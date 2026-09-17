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
from hiver_support.golden import paths  # noqa: E402
from hiver_support.golden.lock import (  # noqa: E402
    GoldenLockError,
    freeze_golden,
    golden_labels_hash,
    validate_golden_set,
    verify_lock,
)
from hiver_support.golden.store import (  # noqa: E402
    load_effective_annotations,
    read_candidates,
    read_retractions,
)
from hiver_support.taxonomy import TAXONOMY  # noqa: E402

GOLDEN_DIR = ROOT / "data" / "golden"
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


def _v2_identity_proof(v2_candidates, annotations) -> dict:
    """Prove v2 is v1 with PII re-masked and nothing else: same examples, same labels."""
    from hiver_support.data.pii import mask_pii

    v1_candidates = read_candidates(paths.require_local_candidates("v1"))
    v1_lock = json.loads(paths.LOCK["v1"].read_text(encoding="utf-8"))
    verify_lock(v1_lock, v1_candidates, annotations)
    if [c.pair_id for c in v1_candidates] != [c.pair_id for c in v2_candidates]:
        raise GoldenLockError("v2 does not contain exactly the v1 examples in the same order")
    changed = []
    for a, b in zip(v1_candidates, v2_candidates):
        same_meta = (a.conversation_id, a.customer_tweet_id, a.created_at) == (b.conversation_id, b.customer_tweet_id, b.created_at)
        if not same_meta or len(a.context) != len(b.context):
            raise GoldenLockError(f"{a.pair_id}: v2 differs from v1 outside the text")
        if mask_pii(a.customer_message).text != b.customer_message or any(
            mask_pii(x.text).text != y.text or x.author_role != y.author_role or x.created_at != y.created_at
            for x, y in zip(a.context, b.context)
        ):
            raise GoldenLockError(f"{a.pair_id}: v2 text is not v1 text re-masked with mask_pii")
        if a.to_dict() != b.to_dict():
            changed.append(a.pair_id)
    labels_v1 = golden_labels_hash(v1_candidates, annotations)
    if labels_v1 != golden_labels_hash(v2_candidates, annotations):
        raise GoldenLockError("v1 and v2 label hashes differ")
    return {
        "v1_lock": "data/golden/GOLDEN_LOCK.json",
        "v1_content_sha256": v1_lock["content_sha256"],
        "v1_labels_sha256": labels_v1,
        "labels_identical": True,
        "examples_identical": True,
        "records_with_text_changes": changed,
        "text_change": "PII re-masking only: mask_pii (v2) applied to each v1 text reproduces v2 exactly",
        "evaluation_boundary": "The reported evaluation used v1. v2 is for future evaluations.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="validate only, write nothing")
    parser.add_argument(
        "--skip-leakage", action="store_true", help="skip corpus guards (needs the dataset)"
    )
    parser.add_argument(
        "--version", choices=paths.VERSIONS, default="v1",
        help="v1 = the evaluated set (already frozen, immutable); v2 = corrected PII masking",
    )
    args = parser.parse_args()

    candidate_file = paths.require_local_candidates(args.version)
    lock_file = paths.LOCK[args.version]
    if lock_file.exists() and not args.check:
        raise SystemExit(
            f"{lock_file.relative_to(ROOT)} already exists. A lock is immutable: it certifies the "
            f"set that was evaluated. Validate with --check; never re-freeze over it."
        )

    candidates = read_candidates(candidate_file)
    # Effective: retracted labels are excluded from gold but stay in the log for audit.
    annotations = load_effective_annotations(ANNOTATIONS)
    retractions = read_retractions(ANNOTATIONS)
    report = validate_golden_set(candidates, annotations)

    print(f"Frozen taxonomy {TAXONOMY.version} ({TAXONOMY.frozen_hash[:16]}...)")
    print(f"Protocol: docs/GOLDEN_SET.md\n")
    print("=" * 74)
    print("GOLDEN SET VALIDATION")
    print("=" * 74)
    for key, value in report.counts.items():
        print(f"  {key:<32} {value}")
    if retractions:
        print(f"  {'retracted (kept, not counted)':<32} {len(retractions)}")

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
    lock["retractions"] = [r.to_dict() for r in retractions]
    lock["retraction_count"] = len(retractions)
    lock["golden_set_version"] = args.version
    lock["candidates_file"] = "data/golden/candidates.jsonl (text-free; full text via scripts/materialize_text.py --golden)"
    lock["labels_sha256"] = golden_labels_hash(candidates, annotations)
    if args.version == "v2":
        lock["derived_from_v1"] = _v2_identity_proof(candidates, annotations)
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

    lock_file.write_bytes((json.dumps(lock, indent=2, ensure_ascii=False) + "\n").encode("utf-8"))
    print("\n" + "=" * 74)
    print(f"FROZEN. content_sha256 = {lock['content_sha256']}")
    print("=" * 74)
    print(f"  wrote {lock_file}")
    print(
        "\n  The golden set is now EVALUATION-ONLY. No model may be trained on it and no\n"
        "  threshold fitted on it; tests/test_golden_lock.py enforces that no training or\n"
        "  agent module can import it at all."
    )


if __name__ == "__main__":
    main()
