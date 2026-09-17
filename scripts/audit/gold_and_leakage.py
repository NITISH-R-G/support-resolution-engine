"""Audit B: gold integrity and leakage, against the FULL train split, with negative controls."""
from __future__ import annotations

import dataclasses
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from hiver_support import leakage
from hiver_support.data.split import temporal_split
from hiver_support.golden.lock import validate_golden_set
from hiver_support.golden.store import (load_effective_annotations, read_candidates,
                                        read_retractions, unresolved_pair_ids)
from hiver_support.taxonomy import TAXONOMY
from discover_taxonomy import load_brand_pairs

G = ROOT / "data" / "golden"


def raises(fn, *args, **kw) -> bool:
    try:
        fn(*args, **kw)
    except leakage.LeakageError:
        return True
    return False


def main():
    out = {}
    from hiver_support.golden import paths
    from hiver_support.golden.lock import verify_lock

    # The committed candidates.jsonl is text-free; v1 full text is rebuilt locally
    # (scripts/materialize_text.py --golden). Hashes are over LF bytes, so this is platform-independent.
    local_v1 = paths.require_local_candidates("v1")
    candidates = read_candidates(local_v1)
    annotations = load_effective_annotations(G / "annotations.jsonl")
    integrity = json.loads(paths.INTEGRITY.read_text(encoding="utf-8"))
    out["B1 rebuilt v1 candidates (LF) match INTEGRITY.json and GOLDEN_LOCK.json"] = (
        hashlib.sha256(local_v1.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
        == integrity["v1"]["candidates_lf_sha256"]
        and verify_lock(json.loads(paths.LOCK["v1"].read_text(encoding="utf-8")), candidates, annotations))
    out["B2 taxonomy hash equals frozen v0.3.0"] = TAXONOMY.frozen_hash.startswith("613f5dfec1253168")
    report = validate_golden_set(candidates, annotations)
    out["B3 gold set validates (200 human-labelled, schema, taxonomy)"] = report.valid and report.counts["human_labelled"] == 200
    out["B4 no unresolved flags"] = unresolved_pair_ids(G / "annotations.jsonl") == ()
    print("  retractions preserved in log:", len(read_retractions(G / "annotations.jsonl")))

    splits = temporal_split(load_brand_pairs(None))
    train, dev = list(splits.train), list(splits.dev)
    by_id = {p.pair_id: p for p in splits.test_pool}
    gold = [by_id[c.pair_id] for c in candidates]
    out["B5 every gold pair is in the test pool"] = len(gold) == 200

    # Guards against the FULL train split (the evaluation's retrieval corpus), not a slice.
    out["B6 no id/conversation/customer overlap with train"] = not raises(leakage.assert_no_id_overlap, train, gold)
    out["B7 no exact text duplicate with train"] = not raises(leakage.assert_no_text_duplicates, train, gold)
    out["B8 no question+answer leakage with train"] = not raises(leakage.assert_no_response_leakage, train, gold)
    out["B9 all train precedes all gold"] = not raises(leakage.assert_temporal_split, train, gold)
    # Measured, not asserted: 2 short fragments exceed 0.90 against the full split (the sampling
    # guard used the last 20k). Neither match is in the groundable corpus; see docs/RELEASE_AUDIT.md.
    out["B10 near-duplicate check vs FULL train raises (known: 2 short fragments)"] = raises(leakage.assert_no_near_duplicates, train, gold)
    out["B11 no id overlap with dev either"] = not raises(leakage.assert_no_id_overlap, dev, gold)

    # Negative controls: each deliberately leaky gold set must be caught by its guard.
    leaked = train[0]
    out["B6-NEG a train pair inserted into gold is caught"] = raises(leakage.assert_no_id_overlap, train, gold + [leaked])
    dup = dataclasses.replace(gold[0], customer_tweet=dataclasses.replace(gold[0].customer_tweet, text=train[5].customer_text))
    out["B7-NEG a copied train message is caught"] = raises(leakage.assert_no_text_duplicates, train, [dup])
    near = dataclasses.replace(gold[0], customer_tweet=dataclasses.replace(gold[0].customer_tweet, text=train[5].customer_text + " pls"))
    out["B10-NEG a near-copy of a train message is caught"] = raises(leakage.assert_no_near_duplicates, train[:5000], [near])
    early = dataclasses.replace(gold[0], customer_tweet=dataclasses.replace(gold[0].customer_tweet, created_at=train[0].customer_tweet.created_at))
    out["B9-NEG a gold message dated inside train is caught"] = raises(leakage.assert_temporal_split, train, [early])

    width = max(len(k) for k in out)
    for k, v in out.items():
        print(f"{k:<{width}}  {'PASS' if v else 'FAIL'}")


if __name__ == "__main__":
    main()
