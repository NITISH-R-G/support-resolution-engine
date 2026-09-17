"""Rewrite git history so no commit contains tweet text. Operates on a SEPARATE clone only.

Release remediation removed tweet text from the current tree. This applies the same rules to
every blob in history and to commit messages, keeping every commit (message, author, date and
order). It never touches the working repository and never pushes.

Rules, per file path (bytes in, bytes out):
    data/golden/candidates.jsonl            any version holding text -> the text-free file from the
                                            remediation commit
    reports/golden_eval/{predictions,judge,judge_claude_partial_402}.jsonl
                                            per row: text fields -> sha256 (hiver_support.golden.textfree)
    data/golden/{suggestions,annotations}.jsonl
                                            per row: pre-annotator rationale -> sha256
    reports/llm_calls.jsonl                 rows holding prompt/response text: those fields removed,
                                            the same note as the redaction at HEAD
    src/hiver_support/taxonomy.py           codebook example text literals -> redaction markers (AST)
    every other text file                   shared detector (scripts/audit/tweet_text.py), key-based
                                            report redaction, failure_analysis quote rule, and the
                                            exact manual replacements made at HEAD (MANUAL)

Usage (on a clone made for the purpose):
    python scripts/audit/rewrite_history_text.py --repo PATH_TO_CLONE --clean-candidates-from REF
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "scripts" / "audit"))

from hiver_support.agent.retrieval import _DEFLECTION_TAIL_RE  # noqa: E402
from hiver_support.golden.textfree import redact_judgement, redact_prediction, sha256_text  # noqa: E402
from remediate_repository_text import redact_quoted_examples, redact_report_keys  # noqa: E402
from tweet_text import TweetIndex, marker, scan_bytes  # noqa: E402

LLM_CALLS_NOTE = "prompt/response text removed in release audit; logging policy stores hashes only"

# Exact replacements made by hand at HEAD, applied to every historical version of these files.
# The old strings ARE dataset quotes, so they live in a gitignored local file, never in this
# script: data/local/rewrite_replacements.json ({"manual": {path: [[old, new], ...]},
# "message_replacements": [[old, new], ...]}). The script refuses to run without it.
REPLACEMENTS_FILE = ROOT / "data" / "local" / "rewrite_replacements.json"


def load_replacements() -> tuple[dict[str, list[tuple[str, str]]], list[tuple[bytes, bytes]]]:
    if not REPLACEMENTS_FILE.exists():
        raise SystemExit(f"{REPLACEMENTS_FILE} is required (local only; it holds the original quotes)")
    data = json.loads(REPLACEMENTS_FILE.read_text(encoding="utf-8"))
    manual = {path: [tuple(pair) for pair in pairs] for path, pairs in data["manual"].items()}
    messages = [(old.encode("utf-8"), new.encode("utf-8")) for old, new in data["message_replacements"]]
    return manual, messages


MANUAL: dict[str, list[tuple[str, str]]] = {}
MESSAGE_REPLACEMENTS: list[tuple[bytes, bytes]] = []


class Redactor:
    def __init__(self, clean_candidates: bytes) -> None:
        self.index = TweetIndex()
        self.clean_candidates = clean_candidates
        self.changed: dict[str, int] = {}

    # ---------------------------------------------------------------- structured JSONL forms
    @staticmethod
    def _jsonl(data: bytes, transform) -> bytes:
        lines = data.replace(b"\r\n", b"\n").decode("utf-8").split("\n")
        out = []
        for line in lines:
            if not line.strip():
                out.append(line)
                continue
            out.append(json.dumps(transform(json.loads(line)), ensure_ascii=False))
        return "\n".join(out).encode("utf-8")

    @staticmethod
    def _prediction(row: dict) -> dict:
        return redact_prediction(row, _DEFLECTION_TAIL_RE) if "message" in row else row

    @staticmethod
    def _judgement(row: dict) -> dict:
        return redact_judgement(row) if "rationale" in row else row

    @staticmethod
    def _rationale(row: dict) -> dict:
        def hashed(target: dict) -> dict:
            return {("rationale_sha256" if k == "rationale" else k): (sha256_text(v) if k == "rationale" else v)
                    for k, v in target.items()}

        if "rationale" in row and "pair_id" in row and "model" in row:
            return hashed(row)
        if isinstance(row.get("model_suggestion"), dict) and "rationale" in row["model_suggestion"]:
            return {k: (hashed(v) if k == "model_suggestion" else v) for k, v in row.items()}
        return row

    @staticmethod
    def _llm_call(row: dict) -> dict:
        if "prompt" not in row and "response" not in row:
            return row
        out = {k: v for k, v in row.items() if k not in ("prompt", "response")}
        out["redacted"] = LLM_CALLS_NOTE
        return out

    # --------------------------------------------------------------------------- taxonomy.py
    def _taxonomy_source(self, data: bytes) -> bytes:
        source = data.decode("utf-8")
        try:
            tree = ast.parse(source)
        except SyntaxError:
            return scan_bytes("x.py", data, self.index, redact=True)[1]
        lines = source.splitlines(keepends=True)
        offsets = [0]
        for line in lines:
            offsets.append(offsets[-1] + len(line))
        spans = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "_ex" and len(node.args) == 3:
                text_node, pair_node = node.args[1], node.args[0]
                if isinstance(text_node, ast.Constant) and isinstance(text_node.value, str):
                    start = offsets[text_node.lineno - 1] + text_node.col_offset
                    end = offsets[text_node.end_lineno - 1] + text_node.end_col_offset
                    tweet_id = str(pair_node.value).split("__")[0] if isinstance(pair_node, ast.Constant) else "0"
                    spans.append((start, end, repr(marker(tweet_id, text_node.value))))
        for start, end, new in sorted(spans, reverse=True):
            source = source[:start] + new + source[end:]
        return scan_bytes("x.py", source.encode("utf-8"), self.index, redact=True)[1]

    # ------------------------------------------------------------------------------ dispatch
    def redact(self, path: str, data: bytes) -> bytes:
        if b"\0" in data[:4096]:
            return data
        try:
            data.decode("utf-8")
        except UnicodeDecodeError:
            return data
        if path == "data/golden/candidates.jsonl":
            return self.clean_candidates if b"customer_message" in data else data
        if path in ("reports/golden_eval/predictions.jsonl",):
            return self._jsonl(data, self._prediction)
        if path in ("reports/golden_eval/judge.jsonl", "reports/golden_eval/judge_claude_partial_402.jsonl"):
            return self._jsonl(data, self._judgement)
        if path in ("data/golden/suggestions.jsonl", "data/golden/annotations.jsonl"):
            return self._jsonl(data, self._rationale)
        if path == "reports/llm_calls.jsonl":
            return self._jsonl(data, self._llm_call)
        if path == "src/hiver_support/taxonomy.py":
            return self._taxonomy_source(data)
        text = data.decode("utf-8")
        for old, new in MANUAL.get(path, []):
            text = text.replace(old, new).replace(old.replace("\n", "\r\n"), new.replace("\n", "\r\n"))
        data = text.encode("utf-8")
        data = scan_bytes(path, data, self.index, redact=True)[1]
        if path.startswith("reports/") and path.endswith(".json"):
            try:
                document, count = redact_report_keys(json.loads(data.decode("utf-8")),
                                                     probes_are_synthetic=path.startswith("reports/llm_smoke"))
            except json.JSONDecodeError:
                count = 0
            if count:
                indent = 2 if b"\n  " in data[:200] else None
                data = (json.dumps(document, indent=indent, ensure_ascii=False) + "\n").encode("utf-8")
        if path == "reports/golden_eval/failure_analysis.md":
            data = redact_quoted_examples(data.decode("utf-8"))[0].encode("utf-8")
        return data


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True, help="a clone made for the rewrite")
    parser.add_argument("--clean-candidates-from", required=True, help="ref whose data/golden/candidates.jsonl is text-free")
    args = parser.parse_args()

    manual, messages = load_replacements()
    MANUAL.update(manual)
    MESSAGE_REPLACEMENTS.extend(messages)

    import git_filter_repo as fr

    repo = args.repo.resolve()
    if repo == ROOT.resolve():
        raise SystemExit("refusing to rewrite the working repository; pass a separate clone")
    clean = subprocess.run(["git", "show", f"{args.clean_candidates_from}:data/golden/candidates.jsonl"],
                           cwd=repo, capture_output=True, check=True).stdout
    redactor = Redactor(clean)
    cache: dict[tuple[bytes, bytes], bytes] = {}
    stats = {"blobs_seen": 0, "blobs_rewritten": 0, "paths_rewritten": {}}

    def file_info_callback(filename, mode, blob_id, value):
        key = (filename, blob_id)
        if key not in cache:
            stats["blobs_seen"] += 1
            original = value.get_contents_by_identifier(blob_id)
            path = filename.decode("utf-8", "replace")
            new = redactor.redact(path, original)
            if new != original:
                stats["blobs_rewritten"] += 1
                stats["paths_rewritten"][path] = stats["paths_rewritten"].get(path, 0) + 1
                cache[key] = value.insert_file_with_contents(new)
            else:
                cache[key] = blob_id
        return (filename, mode, cache[key])

    def message_callback(message):
        for old, new in MESSAGE_REPLACEMENTS:
            message = message.replace(old, new)
        return message

    options = fr.FilteringOptions.parse_args(["--force", "--source", str(repo), "--target", str(repo)])
    filterer = fr.RepoFilter(options, file_info_callback=file_info_callback, message_callback=message_callback)
    filterer.run()
    git_dir = repo / ".git" if (repo / ".git").is_dir() else repo
    (git_dir / "text_rewrite_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
