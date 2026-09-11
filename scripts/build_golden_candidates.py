"""Draw the golden candidate set from the held-out test pool. Writes NO labels.

This is the first and only time the test pool is read. It implements ``docs/GOLDEN_SET.md``,
which was frozen before this script drew anything, and it produces three artifacts:

    data/golden/candidates.jsonl      200 UNLABELED examples - what an annotator reads
    data/golden/sampling_frame.jsonl  stratum, inclusion probability, weight - NOT shown
    data/golden/manifest.json         seed, git sha, taxonomy hash, per-stratum counts

**The candidate file contains no labels and no brand replies.** That is structural: the record
type has no field for either. The sampling frame is kept in a separate file precisely because
the stratum and the weak label are the hints that would anchor an annotator.

Every leakage guard runs against the real train corpus *before* anything is written, and each
raises. A candidate that overlapped the retrieval corpus would let the agent answer an
evaluation question with the evaluation answer.

Usage:
    python scripts/build_golden_candidates.py
    python scripts/build_golden_candidates.py --size 200 --dry-run
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from hiver_support import leakage  # noqa: E402
from hiver_support.data.split import temporal_split  # noqa: E402
from hiver_support.golden.sampling import (  # noqa: E402
    GOLDEN_SIZE,
    RESERVOIR_SIZE,
    SEED,
    candidates_sha256,
    filter_eligible,
    sample_candidates,
)
from hiver_support.golden.store import write_candidates  # noqa: E402
from hiver_support.taxonomy import TAXONOMY  # noqa: E402
from discover_taxonomy import load_brand_pairs  # noqa: E402

GOLDEN_DIR = ROOT / "data" / "golden"
CANDIDATES = GOLDEN_DIR / "candidates.jsonl"
FRAME = GOLDEN_DIR / "sampling_frame.jsonl"
MANIFEST = GOLDEN_DIR / "manifest.json"

# The near-duplicate guard is O(corpus x golden) in a dense matrix, so it runs against a
# bounded, chronologically adjacent slice of train rather than all 800k pairs. Adjacency is
# the point: near-duplicates cluster in time, so the slice nearest the boundary is where a
# paraphrase would actually be found.
NEAR_DUPLICATE_CORPUS = 20_000


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", type=int, default=GOLDEN_SIZE)
    parser.add_argument("--reservoir", type=int, default=RESERVOIR_SIZE)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument(
        "--dry-run", action="store_true", help="sample and report, write nothing"
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="replace an existing candidate file"
    )
    args = parser.parse_args()

    started = time.time()
    print(f"Frozen taxonomy {TAXONOMY.version} ({TAXONOMY.frozen_hash[:16]}...)")
    print(f"Protocol: docs/GOLDEN_SET.md  seed={args.seed}\n")

    print("Loading and splitting AppleSupport pairs ...")
    splits = temporal_split(load_brand_pairs(None))
    train, test_pool = list(splits.train), list(splits.test_pool)
    print(f"  train {len(train):,} | dev {len(splits.dev):,} | test pool {len(test_pool):,}")

    # Eligibility is decided BEFORE the draw. A test-pool message whose normalised text
    # already appears in train would trip assert_no_text_duplicates, and the guard is right:
    # removing such examples after seeing the draw would be picking a result.
    print("\nApplying the pre-registered eligibility filter ...")
    adjacent = train[-NEAR_DUPLICATE_CORPUS:]
    eligible, eligibility = filter_eligible(
        test_pool, train, near_duplicate_corpus=adjacent
    )
    print(
        f"  {eligibility['excluded']:,} of {len(test_pool):,} excluded "
        f"({eligibility['excluded_rate']:.2%}) - "
        f"{eligibility['excluded_exact_duplicate']:,} exact, "
        f"{eligibility['excluded_near_duplicate']:,} near-duplicate"
    )
    print(f"  by stratum: {eligibility['excluded_by_stratum']}")

    print(f"\nSampling {args.size} candidates from {len(eligible):,} eligible pairs ...")
    result = sample_candidates(
        eligible, size=args.size, reservoir=args.reservoir, seed=args.seed
    )
    chosen_ids = {c.pair_id for c in result.candidates}
    chosen_pairs = [p for p in test_pool if p.pair_id in chosen_ids]

    print("\nRunning leakage guards against the real train corpus (each raises) ...")
    leakage.assert_no_id_overlap(train, chosen_pairs)
    print("  id overlap (pair / conversation / customer)  OK")
    leakage.assert_no_text_duplicates(train, chosen_pairs)
    print("  normalised-text duplicates                   OK")
    leakage.assert_no_response_leakage(train, chosen_pairs)
    print("  response leakage (question + answer)         OK")
    leakage.assert_temporal_split(train, chosen_pairs)
    print("  temporal ordering                            OK")
    leakage.assert_no_near_duplicates(adjacent, chosen_pairs)
    print(f"  near-duplicates (vs {len(adjacent):,} adjacent train pairs)   OK")

    strata = Counter(r.stratum for r in result.records)
    components = Counter(r.component for r in result.records)
    print("\n" + "=" * 74)
    print("CANDIDATE SET COMPOSITION - all examples are UNLABELED")
    print("=" * 74)
    for component, count in sorted(components.items()):
        print(f"  {component:<12} {count:>4}")
    print()
    for stratum, count in strata.most_common():
        print(f"  {stratum:<18} {count:>4}")
    shortfall = result.manifest["strata"]["shortfall"]
    if shortfall:
        print(f"\n  quota shortfalls (redistributed, recorded): {shortfall}")

    lengths = [len(c.customer_message.split()) for c in result.candidates]
    with_context = sum(1 for c in result.candidates if c.context)
    print(f"\n  median message length: {sorted(lengths)[len(lengths) // 2]} words")
    print(f"  examples with prior thread context: {with_context}")

    print("\n" + "=" * 74)
    print("THREE EXAMPLES EXACTLY AS AN ANNOTATOR WILL SEE THEM")
    print("=" * 74)
    for candidate in result.candidates[:3]:
        safe = candidate.customer_message.encode("ascii", "replace").decode()
        print(f"\n  [{candidate.pair_id}] {candidate.created_at:%Y-%m-%d %H:%M}")
        print(f"  {safe[:200]}")
        print(f"  label_status = {candidate.label_status.value}")

    if args.dry_run:
        print("\n--dry-run: nothing written.")
        return

    GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
    file_hash = write_candidates(CANDIDATES, result.candidates, overwrite=args.overwrite)
    FRAME.write_text(
        "\n".join(json.dumps(r.to_dict(), ensure_ascii=False) for r in result.records) + "\n",
        encoding="utf-8",
    )

    manifest = dict(result.manifest)
    manifest["candidates_file"] = str(CANDIDATES.relative_to(ROOT)).replace("\\", "/")
    manifest["candidates_sha256"] = file_hash
    manifest["content_sha256"] = candidates_sha256(result.candidates)
    manifest["leakage_guards_run_against_real_train_corpus"] = [
        "assert_no_id_overlap",
        "assert_no_text_duplicates",
        "assert_no_response_leakage",
        "assert_temporal_split",
        "assert_no_near_duplicates",
    ]
    manifest["eligibility_filter"] = eligibility
    manifest["train_pairs"] = len(train)
    manifest["test_pool_pairs"] = len(test_pool)
    manifest["elapsed_seconds"] = round(time.time() - started, 1)
    MANIFEST.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", "utf-8")

    print(f"\nWrote {CANDIDATES}  (sha256 {file_hash[:16]}...)")
    print(f"Wrote {FRAME}")
    print(f"Wrote {MANIFEST}")
    print(
        "\nThis is NOT a golden set yet. It is 200 unlabelled examples.\n"
        "It becomes a golden set only when a human runs:\n"
        "    python scripts/annotate_golden.py --annotator <your-name>"
    )


if __name__ == "__main__":
    main()
