"""The LLM provider layer: configuration, retries, caching, cost, and secret hygiene.

No test here makes a network call. Every provider exposes a single ``_post`` seam that the
transport lives behind, so retry policy, response parsing, token accounting and redaction are
all exercised against constructed responses — including the malformed and hostile ones a real
endpoint eventually returns.

Three behaviours are treated as safety properties rather than conveniences:

* **A missing key fails loudly.** A provider that silently degrades to a plausible string
  would let an unconfigured system produce output that looks like a model wrote it.
* **A malformed response is an error, not an empty reply.** Returning ``""`` on a parse
  failure turns a provider outage into a silent change in agent behaviour.
* **Secrets never reach a log line.** The log is meant to be readable and shareable; one
  leaked key makes it neither.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from hiver_support.agent.llm import (
    CachedProvider,
    LLMConfig,
    LLMNotConfiguredError,
    LLMResponseError,
    LLMResult,
    LLMTransientError,
    NullProvider,
    OpenAICompatibleProvider,
    build_provider,
    estimate_cost_usd,
    redact,
)

KEY = "sk-or-v1-abcdef0123456789abcdef0123456789"


def a_response(text: str = "Try restarting the device.", **overrides) -> dict:
    payload = {
        "id": "gen-1",
        "model": "test/model",
        "choices": [{"message": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 120, "completion_tokens": 30},
    }
    payload.update(overrides)
    return payload


class FakeTransport:
    """Stands in for the HTTP call. Records what it was sent."""

    def __init__(self, *responses):
        self.responses = list(responses)
        self.calls: list[dict] = []

    def __call__(self, payload: dict, headers: dict) -> dict:
        self.calls.append({"payload": payload, "headers": headers})
        outcome = self.responses.pop(0) if self.responses else a_response()
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def a_provider(*responses, **config_overrides) -> OpenAICompatibleProvider:
    settings = {
        "provider": "openrouter",
        "model": "test/model",
        "api_key": KEY,
        "base_url": "https://example.invalid/api/v1",
        "max_retries": 3,
    }
    settings.update(config_overrides)
    provider = OpenAICompatibleProvider(LLMConfig(**settings), sleep=lambda _seconds: None)
    provider._post = FakeTransport(*responses)
    return provider


class TestConfiguration:
    def test_config_reads_provider_and_model_from_the_environment(self):
        config = LLMConfig.from_env(
            {"LLM_PROVIDER": "openrouter", "LLM_MODEL": "x/y", "OPENROUTER_API_KEY": KEY}
        )
        assert config.provider == "openrouter"
        assert config.model == "x/y"
        assert config.api_key == KEY

    def test_no_source_change_is_needed_to_supply_a_key(self):
        # The key arrives by environment only. Nothing reads a literal from the repository.
        config = LLMConfig.from_env({"LLM_PROVIDER": "openrouter", "OPENROUTER_API_KEY": KEY})
        assert config.api_key == KEY

    def test_a_generic_llm_api_key_also_works_for_other_endpoints(self):
        config = LLMConfig.from_env({"LLM_PROVIDER": "openai_compatible", "LLM_API_KEY": "abc"})
        assert config.api_key == "abc"

    def test_an_empty_environment_yields_the_null_provider(self):
        assert LLMConfig.from_env({}).provider == "null"

    def test_numeric_settings_are_parsed_and_defaulted(self):
        config = LLMConfig.from_env(
            {"LLM_TIMEOUT_SECONDS": "12.5", "LLM_MAX_RETRIES": "5", "LLM_MAX_TOKENS": "900"}
        )
        assert (config.timeout_seconds, config.max_retries, config.max_tokens) == (12.5, 5, 900)

    def test_a_malformed_numeric_setting_names_itself_rather_than_defaulting(self):
        with pytest.raises(LLMNotConfiguredError, match="LLM_TIMEOUT_SECONDS"):
            LLMConfig.from_env({"LLM_TIMEOUT_SECONDS": "soon"})

    def test_generation_is_deterministic_by_default(self):
        # Temperature 0 and a fixed seed, so a cache miss during evaluation is reproducible.
        config = LLMConfig.from_env({})
        assert config.temperature == 0.0
        assert config.seed is not None

    def test_openrouter_gets_a_default_base_url_but_it_stays_overridable(self):
        assert "openrouter.ai" in LLMConfig.from_env({"LLM_PROVIDER": "openrouter"}).base_url
        custom = LLMConfig.from_env(
            {"LLM_PROVIDER": "openai_compatible", "LLM_BASE_URL": "https://local.invalid/v1"}
        )
        assert custom.base_url == "https://local.invalid/v1"

    def test_config_never_reveals_the_key_in_its_repr(self):
        config = LLMConfig(provider="openrouter", model="x", api_key=KEY)
        assert KEY not in repr(config)


class TestProviderSubstitution:
    def test_the_default_provider_is_null(self):
        assert isinstance(build_provider(LLMConfig.from_env({})), NullProvider)

    def test_the_null_provider_refuses_loudly_instead_of_inventing_a_reply(self):
        with pytest.raises(LLMNotConfiguredError, match="No LLM provider"):
            NullProvider().generate("hello")

    def test_an_openrouter_provider_is_built_when_a_key_is_present(self):
        provider = build_provider(
            LLMConfig.from_env(
                {"LLM_PROVIDER": "openrouter", "LLM_MODEL": "x/y", "OPENROUTER_API_KEY": KEY}
            )
        )
        assert isinstance(provider, OpenAICompatibleProvider)
        assert provider.name == "openrouter"

    def test_building_a_real_provider_without_a_key_raises_with_instructions(self):
        with pytest.raises(LLMNotConfiguredError, match="OPENROUTER_API_KEY"):
            build_provider(LLMConfig.from_env({"LLM_PROVIDER": "openrouter", "LLM_MODEL": "x"}))

    def test_an_unknown_provider_name_is_refused(self):
        with pytest.raises(LLMNotConfiguredError, match="unknown provider"):
            build_provider(LLMConfig(provider="telepathy", model="x", api_key="k"))

    def test_the_same_interface_serves_every_provider(self):
        for provider in (NullProvider(), a_provider()):
            assert hasattr(provider, "generate")
            assert hasattr(provider, "name")
            assert hasattr(provider, "model")


class TestRequestShape:
    def test_the_request_carries_the_model_temperature_seed_and_token_cap(self):
        provider = a_provider()
        provider.generate("hello")
        payload = provider._post.calls[0]["payload"]
        assert payload["model"] == "test/model"
        assert payload["temperature"] == 0.0
        assert payload["seed"] is not None
        assert payload["max_tokens"] > 0

    def test_the_api_key_travels_in_the_authorization_header(self):
        provider = a_provider()
        provider.generate("hello")
        assert provider._post.calls[0]["headers"]["Authorization"] == f"Bearer {KEY}"

    def test_structured_output_is_requested_when_asked_for(self):
        provider = a_provider(a_response('{"response": "hi"}'))
        provider.generate("hello", json_mode=True)
        assert provider._post.calls[0]["payload"]["response_format"] == {"type": "json_object"}

    def test_a_seed_of_none_is_omitted_rather_than_sent_as_null(self):
        provider = a_provider(seed=None)
        provider.generate("hello")
        assert "seed" not in provider._post.calls[0]["payload"]


class TestResponseParsing:
    def test_a_well_formed_response_yields_text_and_provenance(self):
        result = a_provider().generate("hello")
        assert isinstance(result, LLMResult)
        assert result.text == "Try restarting the device."
        assert (result.provider, result.model) == ("openrouter", "test/model")
        assert result.finish_reason == "stop"

    def test_token_usage_is_recorded_when_the_provider_reports_it(self):
        result = a_provider().generate("hello")
        assert (result.prompt_tokens, result.completion_tokens) == (120, 30)

    def test_missing_usage_yields_zero_rather_than_a_guess(self):
        result = a_provider(a_response(usage={})).generate("hello")
        assert (result.prompt_tokens, result.completion_tokens) == (0, 0)
        assert result.cost_usd is None

    def test_latency_is_recorded(self):
        assert a_provider().generate("hello").latency_ms >= 0

    @pytest.mark.parametrize(
        "bad",
        [
            {},
            {"choices": []},
            {"choices": [{}]},
            {"choices": [{"message": {}}]},
            {"error": {"message": "model not found"}},
        ],
    )
    def test_a_malformed_response_raises_rather_than_returning_an_empty_reply(self, bad):
        # An empty string here would flow into the agent as an "empty draft" and silently
        # convert a provider outage into a change in escalation behaviour.
        with pytest.raises(LLMResponseError):
            a_provider(bad).generate("hello")

    def test_a_provider_error_payload_is_surfaced_with_its_message(self):
        with pytest.raises(LLMResponseError, match="model not found"):
            a_provider({"error": {"message": "model not found"}}).generate("hello")

    def test_a_refusal_or_empty_content_raises(self):
        with pytest.raises(LLMResponseError, match="empty"):
            a_provider(a_response("   ")).generate("hello")


class TestRetryPolicy:
    def test_a_transient_failure_is_retried_and_can_succeed(self):
        provider = a_provider(LLMTransientError("429 rate limited"), a_response("ok"))
        result = provider.generate("hello")
        assert result.text == "ok"
        assert result.attempts == 2

    def test_retries_are_bounded_and_the_last_error_is_raised(self):
        provider = a_provider(
            LLMTransientError("503"), LLMTransientError("503"), LLMTransientError("503"),
            LLMTransientError("503"), max_retries=2,
        )
        with pytest.raises(LLMTransientError):
            provider.generate("hello")
        assert len(provider._post.calls) == 3  # one attempt plus two retries

    def test_backoff_grows_between_attempts(self):
        slept: list[float] = []
        provider = a_provider(LLMTransientError("429"), LLMTransientError("429"), a_response())
        provider.sleep = slept.append
        provider.generate("hello")
        assert slept == sorted(slept) and len(slept) == 2 and slept[1] > slept[0]

    def test_a_permanent_error_is_not_retried(self):
        # Retrying a bad key or a bad model name burns quota to reproduce the same failure.
        provider = a_provider(LLMResponseError("401 invalid api key"))
        with pytest.raises(LLMResponseError):
            provider.generate("hello")
        assert len(provider._post.calls) == 1

    def test_a_timeout_is_transient_and_retried(self):
        provider = a_provider(LLMTransientError("timed out after 30s"), a_response("ok"))
        assert provider.generate("hello").text == "ok"


class TestCost:
    def test_cost_is_computed_from_configured_prices(self):
        assert estimate_cost_usd(1_000_000, 1_000_000, 0.5, 1.5) == pytest.approx(2.0)

    def test_cost_is_none_rather_than_zero_when_prices_are_unknown(self):
        # Reporting $0.00 for an unpriced model would understate spend in the harness.
        assert estimate_cost_usd(1000, 500, None, None) is None

    def test_a_priced_provider_attaches_cost_to_the_result(self):
        provider = a_provider(price_in_per_mtok=1.0, price_out_per_mtok=2.0)
        result = provider.generate("hello")
        assert result.cost_usd == pytest.approx(120 / 1e6 * 1.0 + 30 / 1e6 * 2.0)

    def test_the_provider_accumulates_totals_across_calls(self):
        provider = a_provider(
            a_response(), a_response(), price_in_per_mtok=1.0, price_out_per_mtok=2.0
        )
        provider.generate("a")
        provider.generate("b")
        assert provider.totals["requests"] == 2
        assert provider.totals["prompt_tokens"] == 240
        assert provider.totals["cost_usd"] == pytest.approx(2 * (120e-6 + 60e-6))

    def test_failures_are_counted_separately_from_successes(self):
        provider = a_provider(LLMResponseError("401"))
        with pytest.raises(LLMResponseError):
            provider.generate("hello")
        assert provider.totals["failures"] == 1
        assert provider.totals["requests"] == 0


class TestSecretRedaction:
    def test_redact_replaces_a_known_secret(self):
        assert KEY not in redact(f"Authorization: Bearer {KEY}", [KEY])

    def test_redact_catches_key_shaped_strings_it_was_not_told_about(self):
        leaked = "oops sk-or-v1-deadbeefdeadbeefdeadbeefdeadbeef0000 leaked"
        assert "sk-or-v1-deadbeef" not in redact(leaked, [])

    def test_redact_leaves_ordinary_text_alone(self):
        assert redact("my iphone battery drains", [KEY]) == "my iphone battery drains"

    def test_redact_handles_an_empty_secret_list_and_empty_text(self):
        assert redact("", []) == ""

    def test_an_error_message_from_the_provider_is_redacted(self):
        provider = a_provider({"error": {"message": f"bad key {KEY}"}})
        with pytest.raises(LLMResponseError) as excinfo:
            provider.generate("hello")
        assert KEY not in str(excinfo.value)


class TestLogging:
    def test_a_log_line_is_written_per_call(self, tmp_path):
        log = tmp_path / "llm.jsonl"
        provider = a_provider(log_path=log)
        provider.generate("hello")
        entries = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()]
        assert len(entries) == 1
        assert entries[0]["provider"] == "openrouter"
        assert entries[0]["model"] == "test/model"

    def test_the_log_never_contains_the_api_key(self, tmp_path):
        log = tmp_path / "llm.jsonl"
        a_provider(log_path=log).generate("hello")
        assert KEY not in log.read_text(encoding="utf-8")

    def test_the_log_records_a_prompt_hash_not_the_prompt_by_default(self, tmp_path):
        # Prompts carry customer text. A hash still identifies the request for debugging and
        # cache analysis without copying customer messages into a second place.
        log = tmp_path / "llm.jsonl"
        a_provider(log_path=log).generate("my iphone battery drains badly")
        raw = log.read_text(encoding="utf-8")
        assert "battery drains badly" not in raw
        assert len(json.loads(raw)["prompt_sha256"]) == 64

    def test_prompt_text_is_logged_only_when_explicitly_enabled(self, tmp_path):
        log = tmp_path / "llm.jsonl"
        a_provider(log_path=log, log_prompts=True).generate("my iphone battery drains badly")
        assert "battery drains badly" in log.read_text(encoding="utf-8")

    def test_the_log_records_tokens_cost_latency_and_attempts(self, tmp_path):
        log = tmp_path / "llm.jsonl"
        a_provider(log_path=log, price_in_per_mtok=1.0, price_out_per_mtok=2.0).generate("hi")
        entry = json.loads(log.read_text(encoding="utf-8"))
        assert entry["prompt_tokens"] == 120
        assert entry["cost_usd"] is not None
        assert "latency_ms" in entry and "attempts" in entry

    def test_a_failure_is_logged_too(self, tmp_path):
        log = tmp_path / "llm.jsonl"
        provider = a_provider(LLMResponseError("401 bad key"), log_path=log)
        with pytest.raises(LLMResponseError):
            provider.generate("hello")
        entry = json.loads(log.read_text(encoding="utf-8"))
        assert entry["ok"] is False
        assert "401" in entry["error"]


class TestCaching:
    def test_a_second_identical_call_is_served_from_cache(self, tmp_path):
        inner = a_provider(a_response("first"), a_response("second"))
        cached = CachedProvider(inner, tmp_path)
        assert cached.generate("hello").text == "first"
        again = cached.generate("hello")
        assert again.text == "first"
        assert again.from_cache is True
        assert len(inner._post.calls) == 1

    def test_cache_hits_and_misses_are_counted(self, tmp_path):
        cached = CachedProvider(a_provider(a_response(), a_response()), tmp_path)
        cached.generate("a")
        cached.generate("a")
        cached.generate("b")
        assert (cached.hits, cached.misses) == (1, 2)

    def test_a_different_prompt_misses(self, tmp_path):
        inner = a_provider(a_response("a"), a_response("b"))
        cached = CachedProvider(inner, tmp_path)
        assert cached.generate("one").text == "a"
        assert cached.generate("two").text == "b"

    def test_a_prompt_version_change_misses_rather_than_reusing_an_old_reply(self, tmp_path):
        # A reply written under different instructions is not a reply to the new prompt.
        first = CachedProvider(a_provider(a_response("old")), tmp_path, prompt_version="v1")
        first.generate("hello")
        second = CachedProvider(a_provider(a_response("new")), tmp_path, prompt_version="v2")
        assert second.generate("hello").text == "new"

    def test_a_different_model_misses(self, tmp_path):
        first = CachedProvider(a_provider(a_response("small")), tmp_path)
        first.generate("hello")
        other = CachedProvider(a_provider(a_response("large"), model="test/other"), tmp_path)
        assert other.generate("hello").text == "large"

    def test_a_cached_result_keeps_the_original_provider_and_model(self, tmp_path):
        cached = CachedProvider(a_provider(a_response(), a_response()), tmp_path)
        cached.generate("hello")
        again = cached.generate("hello")
        assert (again.provider, again.model) == ("openrouter", "test/model")

    def test_a_cached_result_reports_no_new_cost(self, tmp_path):
        cached = CachedProvider(
            a_provider(a_response(), price_in_per_mtok=1.0, price_out_per_mtok=2.0), tmp_path
        )
        cached.generate("hello")
        again = cached.generate("hello")
        assert again.cost_usd == 0.0
        assert again.cached_cost_usd_saved is not None

    def test_a_failed_call_is_not_cached(self, tmp_path):
        inner = a_provider(LLMResponseError("401"), a_response("recovered"))
        cached = CachedProvider(inner, tmp_path)
        with pytest.raises(LLMResponseError):
            cached.generate("hello")
        assert cached.generate("hello").text == "recovered"

    def test_a_corrupt_cache_file_is_treated_as_a_miss_not_a_crash(self, tmp_path):
        cached = CachedProvider(a_provider(a_response("fresh")), tmp_path)
        for path in Path(tmp_path).glob("*.json"):
            path.unlink()
        (Path(tmp_path) / f"{cached._key('hello')}.json").write_text("{oops", encoding="utf-8")
        assert cached.generate("hello").text == "fresh"

    def test_the_cache_file_never_contains_the_api_key(self, tmp_path):
        cached = CachedProvider(a_provider(), tmp_path)
        cached.generate("hello")
        for path in Path(tmp_path).glob("*.json"):
            assert KEY not in path.read_text(encoding="utf-8")


class TestNoNetworkCallHappensWithoutCredentials:
    def test_the_null_provider_makes_no_request_and_names_what_to_set(self):
        with pytest.raises(LLMNotConfiguredError) as excinfo:
            NullProvider().generate("hello")
        message = str(excinfo.value)
        assert "LLM_PROVIDER" in message or "provider" in message

    def test_build_provider_with_an_empty_environment_cannot_call_out(self):
        provider = build_provider(LLMConfig.from_env({}))
        with pytest.raises(LLMNotConfiguredError):
            provider.generate("hello")
