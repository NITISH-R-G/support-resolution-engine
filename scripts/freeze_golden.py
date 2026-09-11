"""Validate and freeze the human-labelled golden set. Refuses anything short of complete.

Run this only after annotation is finished. It checks, in order:

    schema            every required field present, types correct
    completeness      no unlabelled example
    taxonomy          every intent inside frozen v0.3.0, hash bound
    candidate ids     annotations reference real candidates, no duplicates
    provenance        every label HUMAN_LABELED, annotator named
    leakage           no overlap with the retrieval corpus, chronology intact
    hash              content hash over questions AND labels

and only then writes ``data/golden/GOLDEN_LOCK.json``.

**It will not freeze a partial set.** A hash stamped on something still changing looks
authoritative and certifies nothing. Run it early and often: with the set incomplete it prints
exactly what is outstanding and exits non-zero, which makes it a progress check as well as a
gate.

Usage:
    python scripts/freeze_golden.py --check     # validate only, write nothing
    python scripts/freeze_golden.py             # validate, run leakage guards, freeze
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from hiver_support import leakage  # noqa: E402
from hiver_support.golden.lock import (  # noqa: E402
    GoldenLockError,
    freeze_golden,
    validate_golden_set,
)
from hiver_support.golden.store import read_annotations, read_candidates  # noqa: E402
from hiver_support.taxonomy import TAXONOMY  # noqa: E402

GOLDEN_DIR = ROOT / "data" / "golden"
CANDIDATES = GOLDEN_DIR / "candidates.jsonl"
ANNOTATIONS = GOLDEN_DIR / "annotations.jsonl"
LOCK = GOLDEN_DIR / "GOLDEN_LOCK.json"

NEAR_DUPLICATE_CORPUS = 20_000


def _run_leakage_guards(candidate_ids: set[str]) -> list[str]:
    """Re-run every guard against the real corpus. Each raises; this reports which passed."""
    from hiver_support.data.split import temporal_split
    from discover_taxonomy import load_brand_pairs

    splits = temporal_split(load_brand_pairs(None))
    train = list(splits.train)
    golden = [p for p in splits.test_pool if p.pair_id in candidate_ids]
    if len(golden) != len(candidate_ids):
        raise GoldenLockError(
            f"only {len(golden)} of {len(candidate_ids)} golden candidates were found in the "
            f"test pool; the split or the candidate file has changed since sampling"
        )

    checks = []
    leakage.assert_no_id_overlap(train, golden)
    checks.append("assert_no_id_overlap")
    leakage.assert_no_text_duplicates(train, golden)
    checks.append("assert_no_text_duplicates")
    leakage.assert_no_response_leakage(train, golden)
    checks.append("assert_no_response_leakage")
    leakage.assert_temporal_split(train, golden)
    checks.append("assert_temporal_split")
    leakage.assert_no_near_duplicates(train[-NEAR_DUPLICATE_CORPUS:], golden)
    checks.append("assert_no_near_duplicates")
    return checks


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="validate only, write nothing")
    parser.add_argument(
        "--skip-leakage", action="store_true", help="skip corpus guards (needs the dataset)"
    )
    args = parser.parse_args()

    candidates = read_candidates(CANDIDATES)
    annotations = read_annotations(ANNOTATIONS)
    report = validate_golden_set(candidates, annotations)

    print(f"Frozen taxonomy {TAXONOMY.version} ({TAXONOMY.frozen_hash[:16]}...)")
    print(f"Protocol: docs/GOLDEN_SET.md\n")
    print("=" * 74)
    print("GOLDEN SET VALIDATION")
    print("=" * 74)
    for key, value in report.counts.items():
        print(f"  {key:<32} {value}")

    if not report.valid:
        print("\n  NOT VALID - the set cannot be frozen yet:")
        for problem in report.problems:
            print(f"    - {problem}")
        print(
            "\n  This is a progress check as well as a gate. Continue with:\n"
            "    python scripts/annotate_golden.py --annotator <your-name>"
        )
        raise SystemExit(1)

    print("\n  schema / completeness / taxonomy / provenance   OK")

    checks: list[str] = []
    if not args.skip_leakage:
        print("\n  Re-running leakage guards against the real corpus (each raises) ...")
        checks = _run_leakage_guards({c.pair_id for c in candidates})
        for check in checks:
            print(f"    {check}   OK")

    lock = freeze_golden(candidates, annotations)
    lock["leakage_guards_rerun_at_freeze"] = checks
    lock["candidates_file"] = str(CANDIDATES.relative_to(ROOT)).replace("\\", "/")
    lock["annotations_file"] = str(ANNOTATIONS.relative_to(ROOT)).replace("\\", "/")

    print("\n" + "=" * 74)
    print("DISTRIBUTIONS")
    print("=" * 74)
    for axis, counts in lock["distributions"].items():
        print(f"\n  {axis}")
        for name, count in counts.items():
            share = count / max(lock["counts"]["human_labelled"], 1)
            print(f"    {name:<28} {count:>4}  ({share:5.1%})")

    if args.check:
        print("\n--check: valid, nothing written.")
        return

    LOCK.write_text(json.dumps(lock, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("\n" + "=" * 74)
    print(f"FROZEN. content_sha256 = {lock['content_sha256']}")
    print("=" * 74)
    print(f"  wrote {LOCK}")
    print(
        "\n  The golden set is now EVALUATION-ONLY. No model may be trained on it and no\n"
        "  threshold fitted on it; tests/test_golden_lock.py enforces that no training or\n"
        "  agent module can import it at all."
    )


if __name__ == "__main__":
    main()
