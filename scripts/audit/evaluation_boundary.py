"""Audit E: the evaluation boundary. Does the hardened system decide exactly what was evaluated?

Three states of the same 800 prediction rows (4 systems x 200 golden examples):

    ORIGINAL   reports/golden_eval/predictions.jsonl at the evaluation commit (git show)
    COMMITTED  the same file at HEAD (re-written offline once, to add score fields)
    HARDENED   a fresh cache-only replay of the CURRENT code, into a temp directory

The replay runs in-process with sockets disabled and the Hugging Face hub forced offline, so a
cache miss or model download cannot silently reach the network. Nothing under reports/ is
overwritten except this audit's own output. The gold set is verified against its lock first.

Negative control: one corrupted row in a copy of the replay must be reported as a difference,
or a report of "0 differences" proves nothing.

Usage:
    python scripts/audit/evaluation_boundary.py
"""
from __future__ import annotations

import copy
import json
import os
import socket
import subprocess
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

EVAL = ROOT / "reports" / "golden_eval"
GOLDEN = ROOT / "data" / "golden"
EVALUATION_COMMIT = "6772707"
OUT_JSON = EVAL / "evaluation_boundary.json"
OUT_MD = EVAL / "evaluation_boundary.md"
ORIGINAL_GENERATOR_FAILURES = json.loads(
    (EVAL / "run_context.json").read_text(encoding="utf-8"))["generator_totals"]["failures"]

# What each comparison means. Fields absent from a state are reported, never guessed.
DECISION = ("escalate",)
ROUTING = ("escalate", "reason")
INTENT = ("intent",)
REPLY = ("reply_sha256",)
# Rows are compared in text-free form (hiver_support.golden.textfree): message, reply and evidence
# text as sha256, plus the derived reply features. A full-text row (a replay, or an original
# artifact that still holds text) is converted first, so every state is comparable.
ALL_OUTPUTS = ("escalate", "reason", "intent", "message_sha256", "reply_sha256", "security_sensitive",
               "context_sufficient", "evidence", "grounding_passed", "derived_text_features",
               "intent_confidence", "retrieval_confidence", "model_confidence")
NOT_OUTPUTS = ("latency_ms", "usage")  # wall-clock and billing, not decisions


def _textfree(row: dict) -> dict:
    from hiver_support.agent.retrieval import _DEFLECTION_TAIL_RE
    from hiver_support.golden.textfree import redact_prediction

    return redact_prediction(row, _DEFLECTION_TAIL_RE) if "message" in row else row


def _rows(text: str) -> dict:
    rows = [_textfree(json.loads(line)) for line in text.splitlines() if line.strip()]
    keyed = {(r["system"], r["pair_id"]): r for r in rows}
    if len(keyed) != len(rows):
        raise SystemExit("duplicate (system, pair_id) rows")
    return keyed


def compare(a: dict, b: dict) -> dict:
    if set(a) != set(b):
        return {"row_sets_equal": False, "only_in_first": len(set(a) - set(b)),
                "only_in_second": len(set(b) - set(a))}
    fields = [f for f in ALL_OUTPUTS if all(f in r for r in a.values()) and all(f in r for r in b.values())]
    missing = [f for f in ALL_OUTPUTS if f not in fields]
    changed_rows, per_field = set(), Counter()
    for key in a:
        for field in fields:
            if a[key][field] != b[key][field]:
                per_field[field] += 1
                changed_rows.add(key)

    def rows_differing(group):
        present = [f for f in group if f in fields]
        if len(present) != len(group):
            return None
        return sum(1 for k in a if any(a[k][f] != b[k][f] for f in group))

    return {
        "row_sets_equal": True,
        "total_rows": len(a),
        "decision_differences": rows_differing(DECISION),
        "routing_differences": rows_differing(ROUTING),
        "intent_differences": rows_differing(INTENT),
        "reply_differences": rows_differing(REPLY),
        "changed_outputs": len(changed_rows),
        "unchanged_outputs": len(a) - len(changed_rows),
        "differences_by_field": {f: per_field.get(f, 0) for f in fields},
        "fields_not_comparable": missing,
        "changed_row_keys": sorted(f"{s}:{p}" for s, p in changed_rows)[:20],
    }


def verify_gold() -> dict:
    from hiver_support.golden.lock import verify_lock
    from hiver_support.golden.store import load_effective_annotations, read_candidates

    from hiver_support.golden import paths

    lock = json.loads((GOLDEN / "GOLDEN_LOCK.json").read_text(encoding="utf-8"))
    verify_lock(lock, read_candidates(paths.require_local_candidates("v1")),
                load_effective_annotations(GOLDEN / "annotations.jsonl"))
    return {"lock_verified": True, "content_sha256": lock["content_sha256"]}


def replay_hardened(masker: str = "v1") -> tuple[dict, dict]:
    """Run the current prediction code from cache only, with the network physically unavailable."""
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    attempts = []

    def no_network(*args, **kwargs):
        attempts.append(args)
        raise OSError("network disabled by evaluation_boundary audit")

    # Import first (ssl subclasses socket.socket), then make every outbound connection fail.
    import evaluate_golden as eg

    real_connect, real_connect_ex = socket.socket.connect, socket.socket.connect_ex
    socket.socket.connect = no_network
    socket.socket.connect_ex = no_network
    try:
        # Control: the block must actually block, or "0 attempts" below is vacuous.
        try:
            socket.create_connection(("1.1.1.1", 443), timeout=2)
            block_works = False
        except OSError as exc:
            block_works = "disabled by evaluation_boundary" in str(exc)
        attempts.clear()
        providers = []
        base = eg.CacheOnlyProvider

        class Recording(base):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                providers.append(self)

        eg.CacheOnlyProvider = Recording
        records, _, _ = eg.predict(None, offline=True, gold_version="v1", masker=masker)
    finally:
        socket.socket.connect, socket.socket.connect_ex = real_connect, real_connect_ex
    totals = [p.totals for p in providers]
    stats = {
        "network_block_control_passed": block_works,
        "network_connection_attempts": len(attempts),
        "cache_misses": sum(getattr(p, "misses", 0) for p in providers),
        "provider_totals": totals,
    }
    keyed = {(r["system"], r["pair_id"]): _textfree(r) for r in records}
    return keyed, stats


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--masker", choices=("v1", "v2"), default="v1",
                        help="v1 = the evaluated configuration; v2 = with the corrected PII masker")
    args = parser.parse_args()
    out_json = OUT_JSON if args.masker == "v1" else EVAL / "evaluation_boundary_masker_v2.json"
    out_md = OUT_MD if args.masker == "v1" else EVAL / "evaluation_boundary_masker_v2.md"
    gold = verify_gold()
    head = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True,
                          text=True).stdout.strip()
    original_text = subprocess.run(
        ["git", "show", f"{EVALUATION_COMMIT}:reports/golden_eval/predictions.jsonl"], cwd=ROOT,
        capture_output=True, text=True, encoding="utf-8", check=True).stdout
    original = _rows(original_text)
    committed = _rows((EVAL / "predictions.jsonl").read_text(encoding="utf-8"))
    code_changes = subprocess.run(
        ["git", "diff", "--stat", EVALUATION_COMMIT, "HEAD", "--", "src", "scripts/evaluate_golden.py"],
        cwd=ROOT,
        capture_output=True, text=True).stdout.strip().splitlines()

    hardened, replay = replay_hardened(args.masker)

    # Negative control: one corrupted routing decision must be reported.
    corrupted = copy.deepcopy(hardened)
    key = next(k for k in sorted(corrupted) if k[0] == "agent_llm")
    corrupted[key]["escalate"] = not corrupted[key]["escalate"]
    control = compare(committed, corrupted)
    negative_control = control["decision_differences"] == 1 and control["changed_outputs"] == 1

    report = {
        "evaluation_commit": EVALUATION_COMMIT,
        "hardened_commit": head,
        "replay_configuration": {"golden_set": "v1", "pii_masker": args.masker},
        "gold": gold,
        "source_changes_since_evaluation": code_changes,
        "replay": replay,
        "original_vs_committed": compare(original, committed),
        "committed_vs_hardened": compare(committed, hardened),
        "original_vs_hardened": compare(original, hardened),
        "negative_control_one_flipped_decision_detected": negative_control,
        "excluded_from_comparison": list(NOT_OUTPUTS),
    }
    out_json.write_bytes((json.dumps(report, indent=2) + "\n").encode("utf-8"))

    c = report["committed_vs_hardened"]
    o = report["original_vs_hardened"]
    lines = [
        "# Evaluation boundary",
        "",
        f"Original evaluated system: commit `{EVALUATION_COMMIT}`. Hardened system: commit `{head}`.",
        f"Gold set verified against `GOLDEN_LOCK.json` (`{gold['content_sha256'][:16]}...`).",
        f"Replay configuration: golden set v1, PII masker {args.masker} "
        f"({'the evaluated configuration' if args.masker == 'v1' else 'post-evaluation hardening: corrected masker'}).",
        "",
        f"Hardened replay: cache only, {replay['network_connection_attempts']} network connection "
        f"attempts, {replay['cache_misses']} cache misses (network block control: "
        f"{'PASS' if replay['network_block_control_passed'] else 'FAIL'}). The original run "
        f"recorded {ORIGINAL_GENERATOR_FAILURES} generator failures; failed responses are never "
        "cached, so each is a cache miss here and escalates as `generator_failed` exactly as "
        "it did originally (the decision comparison below includes those rows).",
        "",
        "| Comparison | Rows | Decision diffs | Routing diffs | Intent diffs | Reply diffs | Changed | Unchanged |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for name, r in (("original vs committed", report["original_vs_committed"]),
                    ("committed vs hardened", c), ("original vs hardened", o)):
        lines.append(f"| {name} | {r['total_rows']} | {r['decision_differences']} | "
                     f"{r['routing_differences']} | {r['intent_differences']} | "
                     f"{r['reply_differences']} | {r['changed_outputs']} | {r['unchanged_outputs']} |")
    lines += [
        "",
        "Compared fields: " + ", ".join(ALL_OUTPUTS) + ". Score fields did not exist at the "
        "evaluation commit, so the original comparisons cover the remaining fields "
        f"(not comparable: {', '.join(o['fields_not_comparable']) or 'none'}). "
        "Latency and token usage are excluded: they are not outputs.",
        "",
        f"Negative control (one flipped decision is reported): "
        f"{'PASS' if negative_control else 'FAIL'}.",
        "",
        "Source changes since the evaluation commit:",
        "",
        "```",
        *code_changes,
        "```",
    ]
    out_md.write_bytes(("\n".join(lines) + "\n").encode("utf-8"))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
