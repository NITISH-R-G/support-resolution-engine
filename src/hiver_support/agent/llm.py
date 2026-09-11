"""Provider abstraction for LLM calls: configuration, retries, caching, cost and redaction.

One interface, several providers. ``OpenAICompatibleProvider`` covers OpenRouter and any
other OpenAI-shaped endpoint including a local one, so swapping models is an environment
variable rather than an edit. Provider-specific handling stops at this file; nothing
downstream knows which model answered.

Four decisions carry the weight:

**The default refuses.** ``NullProvider`` raises a message naming exactly what to configure
rather than returning a plausible string. A silent fallback would let an unconfigured system
emit output that reads as if a model wrote it, which is the most expensive kind of wrong.

**Caching is an evaluation requirement, not an optimisation.** An evaluation that re-queries
on every run is neither reproducible nor free: the same input can yield different text, and a
reviewer re-running the harness pays again. Responses are keyed by (provider, model,
prompt-version, prompt), so a prompt edit correctly misses instead of silently returning a
reply written under different instructions.

**A malformed response raises.** Returning ``""`` would flow into the agent as an empty draft
and quietly convert a provider outage into a change in escalation behaviour — a real failure
disguised as a routing decision.

**Nothing logged can leak a key.** Log lines carry a prompt *hash* by default rather than the
prompt, because prompts contain customer text, and every line is passed through ``redact``.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Callable, Mapping, Sequence

DEFAULT_TIMEOUT_SECONDS = 45.0
DEFAULT_MAX_RETRIES = 3
DEFAULT_MAX_TOKENS = 600
DEFAULT_SEED = 20260911
RETRY_BASE_SECONDS = 1.0
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
USER_AGENT = "support-resolution-engine/1.0 (+https://github.com/NITISH-R-G/support-resolution-engine)"

# Each provider names its own key and its own endpoint. A shared fallback chain was the
# original design and is wrong: with only GROQ_API_KEY set, a request to OpenRouter would
# pick up the Groq key and fail with a 401 that blames the wrong thing.
_PROVIDER_KEY_ENV = {
    "openrouter": "OPENROUTER_API_KEY",
    "groq": "GROQ_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
}
_PROVIDER_BASE_URL = {
    "openrouter": OPENROUTER_BASE_URL,
    "groq": "https://api.groq.com/openai/v1",
}

# Status codes worth retrying: the request was fine, the service was not.
_TRANSIENT_STATUS = frozenset({408, 409, 425, 429, 500, 502, 503, 504})

# Key shapes, so a secret nobody remembered to register still cannot reach a log file.
_SECRET_PATTERNS = (
    re.compile(r"sk-[A-Za-z0-9\-_]{12,}"),
    re.compile(r"\bBearer\s+[A-Za-z0-9\-_.]{12,}"),
    re.compile(r"\b[A-Za-z0-9_\-]{8,}\.[A-Za-z0-9_\-]{12,}\.[A-Za-z0-9_\-]{12,}\b"),
)
REDACTED = "***REDACTED***"


class LLMError(RuntimeError):
    """Base class for provider failures."""


class LLMNotConfiguredError(LLMError):
    """Raised when generation is attempted with no usable provider configuration."""


class LLMResponseError(LLMError):
    """A permanent failure: bad credentials, unknown model, unparseable response.

    Not retried. Re-sending a request that will fail the same way burns quota to reproduce a
    known outcome.
    """


class LLMTransientError(LLMError):
    """A temporary failure worth retrying: timeout, rate limit, 5xx."""


def redact(text: str, secrets: Sequence[str] = ()) -> str:
    """Remove known secrets and key-shaped strings from text destined for a log or an error."""
    if not text:
        return text
    for secret in secrets:
        if secret:
            text = text.replace(secret, REDACTED)
    for pattern in _SECRET_PATTERNS:
        text = pattern.sub(REDACTED, text)
    return text


def estimate_cost_usd(
    prompt_tokens: int,
    completion_tokens: int,
    price_in_per_mtok: float | None,
    price_out_per_mtok: float | None,
) -> float | None:
    """Dollar cost, or ``None`` when the model's prices are unknown.

    ``None`` rather than ``0.0`` deliberately: reporting zero for an unpriced model would
    understate spend in the harness, and a total that silently omits some calls is worse than
    an admitted gap.
    """
    if price_in_per_mtok is None or price_out_per_mtok is None:
        return None
    return (prompt_tokens / 1e6) * price_in_per_mtok + (completion_tokens / 1e6) * price_out_per_mtok


def _as_float(env: Mapping[str, str], key: str, default: float) -> float:
    raw = env.get(key)
    if raw is None or raw == "":
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise LLMNotConfiguredError(f"{key}={raw!r} is not a number") from exc


def _as_int(env: Mapping[str, str], key: str, default: int) -> int:
    return int(_as_float(env, key, float(default)))


def load_dotenv(path: Path | None = None) -> None:
    """Load .env into os.environ without overriding anything already set.

    Lives here rather than in each script so every entry point resolves configuration the
    same way. Values already in the environment win, so an explicit export still overrides
    the file. Nothing is ever printed: a key that reaches stdout is a leaked key.
    """
    env_file = Path(path) if path else Path(__file__).resolve().parents[3] / ".env"
    if not env_file.exists():
        return
    for line in env_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


@dataclass(frozen=True)
class LLMConfig:
    """Everything a provider needs, sourced from the environment. No secret is ever a literal."""

    provider: str = "null"
    model: str = ""
    api_key: str | None = None
    base_url: str = ""
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS
    max_retries: int = DEFAULT_MAX_RETRIES
    temperature: float = 0.0
    seed: int | None = DEFAULT_SEED
    max_tokens: int = DEFAULT_MAX_TOKENS
    price_in_per_mtok: float | None = None
    price_out_per_mtok: float | None = None
    log_path: Path | None = None
    log_prompts: bool = False
    referer: str = "https://github.com/NITISH-R-G/support-resolution-engine"
    app_title: str = "support-resolution-engine"

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> LLMConfig:
        env = os.environ if env is None else env
        provider = (env.get("LLM_PROVIDER") or "null").strip().lower()

        # A named provider reads its own variable, falling back only to the generic
        # LLM_API_KEY. Generic endpoints accept any of them, since there is no better guess.
        specific = _PROVIDER_KEY_ENV.get(provider)
        if specific:
            api_key = env.get(specific) or env.get("LLM_API_KEY") or None
        else:
            api_key = (
                env.get("LLM_API_KEY")
                or env.get("OPENROUTER_API_KEY")
                or env.get("GROQ_API_KEY")
                or env.get("ANTHROPIC_API_KEY")
                or env.get("OPENAI_API_KEY")
                or None
            )
        base_url = env.get("LLM_BASE_URL") or _PROVIDER_BASE_URL.get(provider, "")
        seed_raw = env.get("LLM_SEED", "")
        seed = None if seed_raw.strip().lower() in ("none", "off") else _as_int(
            env, "LLM_SEED", DEFAULT_SEED
        )
        log_path = env.get("LLM_LOG_PATH")

        return cls(
            provider=provider,
            model=(env.get("LLM_MODEL") or "").strip(),
            api_key=api_key,
            base_url=base_url,
            timeout_seconds=_as_float(env, "LLM_TIMEOUT_SECONDS", DEFAULT_TIMEOUT_SECONDS),
            max_retries=_as_int(env, "LLM_MAX_RETRIES", DEFAULT_MAX_RETRIES),
            temperature=_as_float(env, "LLM_TEMPERATURE", 0.0),
            seed=seed,
            max_tokens=_as_int(env, "LLM_MAX_TOKENS", DEFAULT_MAX_TOKENS),
            price_in_per_mtok=(
                _as_float(env, "LLM_PRICE_IN_PER_MTOK", -1.0)
                if env.get("LLM_PRICE_IN_PER_MTOK")
                else None
            ),
            price_out_per_mtok=(
                _as_float(env, "LLM_PRICE_OUT_PER_MTOK", -1.0)
                if env.get("LLM_PRICE_OUT_PER_MTOK")
                else None
            ),
            log_path=Path(log_path) if log_path else None,
            log_prompts=(env.get("LLM_LOG_PROMPTS", "").strip().lower() in ("1", "true", "yes")),
        )

    def __repr__(self) -> str:  # pragma: no cover - trivial, but a leak here is permanent
        return (
            f"LLMConfig(provider={self.provider!r}, model={self.model!r}, "
            f"api_key={'set' if self.api_key else 'unset'}, base_url={self.base_url!r}, "
            f"timeout_seconds={self.timeout_seconds}, max_retries={self.max_retries}, "
            f"temperature={self.temperature}, seed={self.seed})"
        )


@dataclass(frozen=True, slots=True)
class LLMResult:
    """One generation plus everything the harness needs to account for it."""

    text: str
    provider: str
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float | None = None
    latency_ms: float = 0.0
    attempts: int = 1
    finish_reason: str = ""
    from_cache: bool = False
    cached_cost_usd_saved: float | None = None
    prompt_version: str = "v1"

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "provider": self.provider,
            "model": self.model,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "cost_usd": self.cost_usd,
            "latency_ms": round(self.latency_ms, 1),
            "attempts": self.attempts,
            "finish_reason": self.finish_reason,
            "from_cache": self.from_cache,
            "prompt_version": self.prompt_version,
        }


class LLMProvider:
    """The interface every provider implements. Nothing downstream knows which one it holds."""

    name: str = "base"
    model: str = "none"

    def generate(self, prompt: str, *, json_mode: bool = False) -> LLMResult:
        raise NotImplementedError

    def complete(self, prompt: str, **_kwargs) -> str:
        """Text-only convenience for callers that need nothing but the reply."""
        return self.generate(prompt).text

    @property
    def totals(self) -> dict:
        return {"requests": 0, "failures": 0, "prompt_tokens": 0, "completion_tokens": 0,
                "cost_usd": 0.0}


class NullProvider(LLMProvider):
    """The default. Refuses loudly instead of inventing a reply."""

    name = "null"
    model = "none"

    def generate(self, prompt: str, *, json_mode: bool = False) -> LLMResult:
        raise LLMNotConfiguredError(
            "No LLM provider is configured, and this layer will not invent a reply. The agent "
            "runs end-to-end without one using the deterministic evidence-grounded generator. "
            "To enable a real model, copy .env.example to .env and set LLM_PROVIDER, "
            "LLM_MODEL and the matching API key (e.g. OPENROUTER_API_KEY). "
            "See docs/LLM_PROVIDER.md."
        )


class OpenAICompatibleProvider(LLMProvider):
    """OpenRouter, OpenAI, or any other endpoint speaking the chat-completions shape.

    The transport is isolated in ``_post`` so retry policy, parsing, accounting and redaction
    are all exercisable without a network — which is the only way the malformed and hostile
    responses a real endpoint eventually returns get tested at all.
    """

    def __init__(self, config: LLMConfig, sleep: Callable[[float], None] = time.sleep) -> None:
        if not config.api_key:
            expected = _PROVIDER_KEY_ENV.get(config.provider, "LLM_API_KEY")
            raise LLMNotConfiguredError(
                f"MISSING: {expected}\n"
                f"provider {config.provider!r} needs an API key. Set {expected} "
                f"(or LLM_API_KEY) in .env or the environment; never in source."
            )
        if not config.model:
            raise LLMNotConfiguredError(
                f"provider {config.provider!r} needs LLM_MODEL, e.g. "
                f"LLM_MODEL=meta-llama/llama-3.3-70b-instruct"
            )
        self.config = config
        self.name = config.provider
        self.model = config.model
        self.sleep = sleep
        self._totals = {
            "requests": 0, "failures": 0, "prompt_tokens": 0, "completion_tokens": 0,
            "cost_usd": 0.0, "retries": 0,
        }

    @property
    def totals(self) -> dict:
        return dict(self._totals)

    # ------------------------------------------------------------------ transport

    def _headers(self) -> dict:
        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json",
            # Found against the real Groq endpoint: with no User-Agent, urllib advertises
            # "Python-urllib/3.x" and Cloudflare rejects the request with HTTP 403
            # "error code: 1010" (banned browser signature) before it ever reaches the API.
            # Nothing in a constructed fixture could have surfaced this - the transport is
            # exactly the part the fixtures replace.
            "User-Agent": USER_AGENT,
        }
        if self.name == "openrouter":  # noqa: SIM102 - other providers ignore these
            # OpenRouter attributes usage to an app; both are optional and carry no secret.
            headers["HTTP-Referer"] = self.config.referer
            headers["X-Title"] = self.config.app_title
        return headers

    def _post(self, payload: dict, headers: dict) -> dict:  # pragma: no cover - network
        request = urllib.request.Request(
            f"{self.config.base_url.rstrip('/')}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.config.timeout_seconds) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", "replace")[:500]
            message = redact(f"HTTP {exc.code}: {body}", [self.config.api_key or ""])
            if exc.code in _TRANSIENT_STATUS:
                raise LLMTransientError(message) from exc
            raise LLMResponseError(message) from exc
        except urllib.error.URLError as exc:
            raise LLMTransientError(redact(f"network error: {exc.reason}", [])) from exc
        except TimeoutError as exc:
            raise LLMTransientError(
                f"timed out after {self.config.timeout_seconds}s"
            ) from exc

    # ------------------------------------------------------------------ generation

    def _build_payload(self, prompt: str, json_mode: bool) -> dict:
        payload = {
            "model": self.config.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
        }
        if self.config.seed is not None:
            payload["seed"] = self.config.seed
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        return payload

    def _parse(self, body: object) -> tuple[str, str, int, int]:
        secrets = [self.config.api_key or ""]
        if not isinstance(body, dict):
            raise LLMResponseError(f"response was {type(body).__name__}, expected an object")
        if "error" in body:
            detail = body["error"]
            message = detail.get("message") if isinstance(detail, dict) else str(detail)
            raise LLMResponseError(redact(f"provider error: {message}", secrets))

        choices = body.get("choices")
        if not isinstance(choices, list) or not choices:
            raise LLMResponseError(
                redact(f"response has no choices: {json.dumps(body)[:300]}", secrets)
            )
        message = choices[0].get("message") if isinstance(choices[0], dict) else None
        if not isinstance(message, dict) or "content" not in message:
            raise LLMResponseError(
                redact(f"choice has no message content: {json.dumps(choices[0])[:300]}", secrets)
            )
        text = (message.get("content") or "").strip()
        if not text:
            raise LLMResponseError("provider returned empty content")

        usage = body.get("usage") or {}
        return (
            text,
            str(choices[0].get("finish_reason") or ""),
            int(usage.get("prompt_tokens") or 0),
            int(usage.get("completion_tokens") or 0),
        )

    def generate(self, prompt: str, *, json_mode: bool = False) -> LLMResult:
        payload = self._build_payload(prompt, json_mode)
        headers = self._headers()
        started = time.time()
        attempt = 0
        last_error: Exception | None = None

        while attempt <= self.config.max_retries:
            attempt += 1
            try:
                body = self._post(payload, headers)
                text, finish_reason, prompt_tokens, completion_tokens = self._parse(body)
            except LLMTransientError as exc:
                last_error = exc
                if attempt > self.config.max_retries:
                    break
                self._totals["retries"] += 1
                # Exponential backoff, no jitter: an evaluation run should be reproducible in
                # wall-clock terms as well as in output.
                self.sleep(RETRY_BASE_SECONDS * (2 ** (attempt - 1)))
                continue
            except LLMError as exc:
                self._totals["failures"] += 1
                self._log(prompt, None, (time.time() - started) * 1000, attempt, exc)
                raise

            cost = estimate_cost_usd(
                prompt_tokens,
                completion_tokens,
                self.config.price_in_per_mtok,
                self.config.price_out_per_mtok,
            )
            result = LLMResult(
                text=text,
                provider=self.name,
                model=self.model,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                cost_usd=cost,
                latency_ms=(time.time() - started) * 1000,
                attempts=attempt,
                finish_reason=finish_reason,
            )
            self._totals["requests"] += 1
            self._totals["prompt_tokens"] += prompt_tokens
            self._totals["completion_tokens"] += completion_tokens
            self._totals["cost_usd"] += cost or 0.0
            self._log(prompt, result, result.latency_ms, attempt, None)
            return result

        self._totals["failures"] += 1
        self._log(prompt, None, (time.time() - started) * 1000, attempt, last_error)
        raise last_error if last_error else LLMResponseError("exhausted retries without a result")

    # ------------------------------------------------------------------ logging

    def _log(
        self,
        prompt: str,
        result: LLMResult | None,
        latency_ms: float,
        attempts: int,
        error: Exception | None,
    ) -> None:
        path = self.config.log_path
        if not path:
            return
        secrets = [self.config.api_key or ""]
        entry = {
            "timestamp": time.time(),
            "provider": self.name,
            "model": self.model,
            # A hash, not the prompt: prompts carry customer text, and copying it into a
            # second file makes the PII boundary two places instead of one.
            "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
            "prompt_chars": len(prompt),
            "ok": error is None,
            "latency_ms": round(latency_ms, 1),
            "attempts": attempts,
            "prompt_tokens": result.prompt_tokens if result else 0,
            "completion_tokens": result.completion_tokens if result else 0,
            "cost_usd": result.cost_usd if result else None,
            "finish_reason": result.finish_reason if result else "",
            "error": redact(str(error), secrets) if error else None,
        }
        if self.config.log_prompts:
            entry["prompt"] = redact(prompt, secrets)
            entry["response"] = redact(result.text, secrets) if result else None
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with Path(path).open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry, ensure_ascii=False) + "\n")


class AnthropicProvider(LLMProvider):  # pragma: no cover - requires a key to exercise
    """Anthropic Claude, kept so the judge can be a different family from the generator.

    ``leakage.assert_independent_models`` requires exactly that: a judge sharing a family with
    the generator is the contamination this project audits others for.
    """

    def __init__(self, config: LLMConfig) -> None:
        self.config = config
        self.name = "anthropic"
        self.model = config.model or "claude-sonnet-5"
        if not config.api_key:
            raise LLMNotConfiguredError("ANTHROPIC_API_KEY is not set")
        self._totals = {"requests": 0, "failures": 0, "prompt_tokens": 0,
                        "completion_tokens": 0, "cost_usd": 0.0}

    @property
    def totals(self) -> dict:
        return dict(self._totals)

    def generate(self, prompt: str, *, json_mode: bool = False) -> LLMResult:
        from anthropic import Anthropic

        started = time.time()
        message = Anthropic(api_key=self.config.api_key).messages.create(
            model=self.model,
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
            messages=[{"role": "user", "content": prompt}],
        )
        text = "".join(block.text for block in message.content if block.type == "text").strip()
        if not text:
            raise LLMResponseError("provider returned empty content")
        prompt_tokens = getattr(message.usage, "input_tokens", 0)
        completion_tokens = getattr(message.usage, "output_tokens", 0)
        self._totals["requests"] += 1
        self._totals["prompt_tokens"] += prompt_tokens
        self._totals["completion_tokens"] += completion_tokens
        return LLMResult(
            text=text,
            provider=self.name,
            model=self.model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cost_usd=estimate_cost_usd(
                prompt_tokens, completion_tokens,
                self.config.price_in_per_mtok, self.config.price_out_per_mtok,
            ),
            latency_ms=(time.time() - started) * 1000,
        )


class CachedProvider(LLMProvider):
    """Wraps any provider with an on-disk cache keyed by provider, model, prompt and version.

    The key includes the prompt version so an edited prompt misses the cache. Reusing a reply
    written under different instructions would be worse than paying again: it would attribute
    old behaviour to a new prompt, which is unfalsifiable from the outputs alone.
    """

    def __init__(
        self, provider: LLMProvider, cache_dir: Path, prompt_version: str = "v1"
    ) -> None:
        self.provider = provider
        self.name = provider.name
        self.model = provider.model
        self.prompt_version = prompt_version
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.hits = 0
        self.misses = 0
        self.cost_saved_usd = 0.0

    @property
    def totals(self) -> dict:
        totals = dict(self.provider.totals)
        totals["cache_hits"] = self.hits
        totals["cache_misses"] = self.misses
        totals["cost_saved_usd"] = self.cost_saved_usd
        return totals

    def _key(self, prompt: str, json_mode: bool = False) -> str:
        payload = (
            f"{self.provider.name}|{self.provider.model}|{self.prompt_version}|"
            f"{int(json_mode)}|{prompt}"
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def generate(self, prompt: str, *, json_mode: bool = False) -> LLMResult:
        path = self.cache_dir / f"{self._key(prompt, json_mode)}.json"
        if path.exists():
            try:
                stored = json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                # A truncated cache file is a miss, not a crash. Re-paying for one response
                # is cheaper than an aborted evaluation run.
                stored = None
            if stored:
                self.hits += 1
                saved = stored.get("cost_usd")
                self.cost_saved_usd += saved or 0.0
                return LLMResult(
                    text=stored["text"],
                    provider=stored.get("provider", self.name),
                    model=stored.get("model", self.model),
                    prompt_tokens=stored.get("prompt_tokens", 0),
                    completion_tokens=stored.get("completion_tokens", 0),
                    cost_usd=0.0,
                    latency_ms=0.0,
                    finish_reason=stored.get("finish_reason", ""),
                    from_cache=True,
                    cached_cost_usd_saved=saved if saved is not None else 0.0,
                    prompt_version=self.prompt_version,
                )

        self.misses += 1
        result = replace(self.provider.generate(prompt, json_mode=json_mode),
                         prompt_version=self.prompt_version)
        # Only successful calls reach here, so a failure is never cached — an outage must not
        # become a permanent answer.
        path.write_text(
            json.dumps(
                {**result.to_dict(), "prompt_sha256": hashlib.sha256(
                    prompt.encode("utf-8")).hexdigest()},
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        return result


_PROVIDERS: dict[str, Callable[[LLMConfig], LLMProvider]] = {
    "null": lambda _config: NullProvider(),
    "openrouter": OpenAICompatibleProvider,
    "groq": OpenAICompatibleProvider,
    "openai_compatible": OpenAICompatibleProvider,
    "openai": OpenAICompatibleProvider,
    "anthropic": AnthropicProvider,
}


def build_provider(config: LLMConfig) -> LLMProvider:
    """Construct the configured provider. Unknown names raise rather than defaulting."""
    factory = _PROVIDERS.get(config.provider)
    if factory is None:
        raise LLMNotConfiguredError(
            f"unknown provider {config.provider!r}; set LLM_PROVIDER to one of "
            f"{sorted(_PROVIDERS)}"
        )
    return factory(config)
