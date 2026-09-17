"""Rebuild the tweet text the repository deliberately does not store, and verify every hash.

The repository holds ids, labels, edit scripts and sha256 hashes only. This script rebuilds the
text from the local Kaggle download into ``data/local/`` (gitignored). It raises on the first
mismatch rather than writing anything unverified.

    --codebook     29 taxonomy codebook examples, rebuilt from edit scripts. Each must match its
                   hash, and together they must reproduce taxonomy hash 613f5dfe...
    --golden       golden set v1 (frozen v1 masker) and v2 (corrected masker). Every record's
                   text hashes must match, each file's LF hash must match INTEGRITY.json, and
                   each version must verify against its lock.
    --evaluation   the original full-text predictions.jsonl, judge.jsonl and
                   judge_claude_partial_402.jsonl. Requires the local LLM response cache
                   (cache/llm/, never committed): texts come from a cache-only replay of the
                   evaluated configuration (golden v1, masker v1). Each rebuilt file must match
                   ORIGINAL_ARTIFACT_HASHES.json byte for byte.

Usage:
    python scripts/materialize_text.py --codebook --golden
    python scripts/materialize_text.py --evaluation [--replay-dir DIR]
"""
from __future__ import annotations

import argparse
import dataclasses
import glob
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from hiver_support import codebook  # noqa: E402
from hiver_support.data.pii import mask_pii, mask_pii_v1  # noqa: E402
from hiver_support.golden import paths  # noqa: E402
from hiver_support.golden.textfree import (  # noqa: E402
    TextMismatchError,
    restore_judgement,
    restore_prediction,
    verify_candidate,
)

EVAL = ROOT / "reports" / "golden_eval"
MASKERS = {"v1": mask_pii_v1, "v2": mask_pii}


def _pairs():
    from discover_taxonomy import load_brand_pairs

    return load_brand_pairs(None)


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def _jsonl_bytes(rows: list[dict]) -> bytes:
    return ("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n").encode("utf-8")


def materialize_codebook(pairs) -> dict:
    from hiver_support.taxonomy import TAXONOMY

    texts = codebook.rebuild({p.pair_id: p.customer_text for p in pairs})
    intents = tuple(
        dataclasses.replace(i, examples=tuple(dataclasses.replace(e, text=texts[e.pair_id]) for e in i.examples))
        for i in TAXONOMY.intents
    )
    rebuilt = dataclasses.replace(TAXONOMY, intents=intents)
    content = rebuilt.content_hash()
    if content != codebook.frozen_taxonomy_hash():
        raise SystemExit(f"rebuilt codebook gives taxonomy hash {content[:16]}..., not the frozen hash")
    codebook.MATERIALIZED.parent.mkdir(parents=True, exist_ok=True)
    codebook.MATERIALIZED.write_bytes((json.dumps({"texts": texts}, ensure_ascii=False, indent=1) + "\n").encode("utf-8"))
    return {"examples": len(texts), "hashes_verified": len(texts), "taxonomy_hash": content}


def materialize_golden(pairs) -> dict:
    from hiver_support.data.split import temporal_split
    from hiver_support.golden.lock import verify_lock
    from hiver_support.golden.sampling import _to_candidate
    from hiver_support.golden.store import load_effective_annotations, read_candidates, write_candidates

    records = _jsonl(paths.CANDIDATE_RECORDS)
    pool = {p.pair_id: p for p in temporal_split(pairs).test_pool}
    integrity = json.loads(paths.INTEGRITY.read_text(encoding="utf-8"))
    result = {}
    for version in paths.VERSIONS:
        rebuilt = []
        for record in records:
            candidate = _to_candidate(pool[record["pair_id"]], masker=MASKERS[version])
            verify_candidate(record, candidate.to_dict(), version)
            rebuilt.append(candidate)
        digest = write_candidates(paths.local_candidates(version), rebuilt, overwrite=True)
        if digest != integrity[version]["candidates_lf_sha256"]:
            raise SystemExit(f"golden {version}: file hash {digest[:16]}... does not match INTEGRITY.json")
        lock_path = paths.LOCK[version]
        locked = "no lock file"
        if lock_path.exists():
            lock = json.loads(lock_path.read_text(encoding="utf-8"))
            verify_lock(lock, read_candidates(paths.local_candidates(version)),
                        load_effective_annotations(paths.ANNOTATIONS))
            locked = f"verified ({lock['content_sha256'][:16]}...)"
        result[version] = {"records_verified": len(rebuilt), "file_lf_sha256": digest, "lock": locked}
    result["pre_annotator_rationales"] = _restore_rationales(integrity)
    return result


def _restore_rationales(integrity: dict) -> dict:
    """Rebuild suggestions.jsonl and annotations.jsonl with rationale text from the local
    pre-annotator cache and check them against the original file hashes. Labels are unaffected
    either way; this proves nothing was lost."""
    from hiver_support.golden.suggestions import parse_suggestion

    by_hash = {}
    for path in glob.glob(str(ROOT / "cache" / "llm" / "meta-llama_llama-3.3-70b-instruct" / "*.json")):
        try:
            rationale = parse_suggestion("x", json.loads(Path(path).read_text(encoding="utf-8"))["text"], "m", "p").rationale
        except Exception:  # noqa: BLE001 - an unparseable cached response simply contributes nothing
            continue
        by_hash[hashlib.sha256(rationale.encode("utf-8")).hexdigest()] = rationale
    if not by_hash:
        return {"status": "skipped: no local pre-annotator cache (cache/llm/, never committed)"}

    def restore(target: dict) -> dict:
        digest = target["rationale_sha256"]
        if digest not in by_hash:
            raise TextMismatchError("a pre-annotator rationale is not in the local cache")
        return {("rationale" if k == "rationale_sha256" else k): (by_hash[digest] if k == "rationale_sha256" else v)
                for k, v in target.items()}

    report = {}
    for name, path, key in (("suggestions", paths.GOLDEN_DIR / "suggestions.jsonl", None),
                            ("annotations", paths.ANNOTATIONS, "model_suggestion")):
        rows = []
        for row in _jsonl(path):
            if key is None and "rationale_sha256" in row:
                row = restore(row)
            elif key and isinstance(row.get(key), dict) and "rationale_sha256" in row[key]:
                row = {k: (restore(v) if k == key else v) for k, v in row.items()}
            rows.append(row)
        data = _jsonl_bytes(rows)
        expected = integrity["v1"][f"{name}_original_lf_sha256"]
        if hashlib.sha256(data).hexdigest() != expected:
            raise SystemExit(f"rebuilt {name} does not match its original hash")
        target = paths.LOCAL_DIR / "golden" / f"{name}.full.jsonl"
        target.write_bytes(data)
        report[name] = {"rows": len(rows), "file_hash_matches_original": True}
    return report


def _replay(replay_dir: Path | None) -> Path:
    if replay_dir is not None:
        return replay_dir
    target = paths.LOCAL_DIR / "golden_eval_replay_v1"
    env = dict(os.environ)
    # Cache-only replay: no request is made. The providers still need key variables to exist.
    env.setdefault("OPENROUTER_API_KEY", "offline-replay")
    env.setdefault("GROQ_API_KEY", "offline-replay")
    subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "evaluate_golden.py"), "--stage", "all", "--offline",
         "--gold", "v1", "--masker", "v1", "--judge-provider", "groq", "--judge-model", "qwen/qwen3.8-27b",
         "--out", str(target)],
        check=True, env=env, cwd=ROOT,
    )
    return target


def _cached_rationales() -> dict[str, str]:
    """sha256 -> rationale for every judge response in the local cache (both judge runs)."""
    found = {}
    for path in glob.glob(str(ROOT / "cache" / "llm" / "golden_eval_judge" / "**" / "*.json"), recursive=True):
        try:
            text = json.loads(Path(path).read_text(encoding="utf-8"))["text"]
            rationale = json.loads(text)["rationale"]
        except (KeyError, ValueError, TypeError):
            continue
        found[hashlib.sha256(rationale.encode("utf-8")).hexdigest()] = rationale
    return found


def materialize_evaluation(replay_dir: Path | None) -> dict:
    originals = json.loads((EVAL / "ORIGINAL_ARTIFACT_HASHES.json").read_text(encoding="utf-8"))["sha256"]
    replay = _replay(replay_dir)
    replayed = {(r["system"], r["pair_id"]): r for r in _jsonl(replay / "predictions.jsonl")}
    out_dir = paths.local_eval_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    result = {}

    restored = []
    for row in _jsonl(EVAL / "predictions.jsonl"):
        source = replayed[(row["system"], row["pair_id"])]
        restored.append(restore_prediction(
            row, source["message"], source["reply"],
            [(e["customer_text"], e["resolution_text"]) for e in source["evidence"]],
        ))
    data = _jsonl_bytes(restored)
    if hashlib.sha256(data).hexdigest() != originals["predictions.jsonl"]:
        raise SystemExit("rebuilt predictions.jsonl does not match the original artifact hash")
    (out_dir / "predictions.jsonl").write_bytes(data)
    result["predictions.jsonl"] = {"rows_verified": len(restored), "file_hash_matches_original": True}

    # Rationales as parsed by the judge itself in the replay (the cache holds raw responses, some
    # wrapped in fences), plus any cached responses for the partial Claude run.
    rationales = _cached_rationales()
    for row in _jsonl(replay / "judge.jsonl"):
        if row.get("rationale") is not None:
            rationales[hashlib.sha256(row["rationale"].encode("utf-8")).hexdigest()] = row["rationale"]
    for name in ("judge.jsonl", "judge_claude_partial_402.jsonl"):
        rows, missing = [], 0
        for row in _jsonl(EVAL / name):
            digest = row.get("rationale_sha256")
            if digest is not None and digest not in rationales:
                missing += 1
                rows.append(None)
                continue
            rows.append(restore_judgement(row, rationales.get(digest)) if digest is not None else
                        {("rationale" if k == "rationale_sha256" else k): v for k, v in row.items()})
        if missing:
            result[name] = {"rows": len(rows), "rationales_not_in_local_cache": missing,
                            "file_hash_matches_original": False}
            continue
        data = _jsonl_bytes(rows)
        matches = hashlib.sha256(data).hexdigest() == originals[name]
        if not matches:
            raise SystemExit(f"rebuilt {name} does not match the original artifact hash")
        (out_dir / name).write_bytes(data)
        result[name] = {"rows_verified": len(rows), "file_hash_matches_original": True}
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--codebook", action="store_true")
    parser.add_argument("--golden", action="store_true")
    parser.add_argument("--evaluation", action="store_true")
    parser.add_argument("--replay-dir", type=Path, default=None,
                        help="reuse an existing cache-only replay of the evaluated configuration")
    args = parser.parse_args()
    if not (args.codebook or args.golden or args.evaluation):
        parser.error("choose at least one of --codebook, --golden, --evaluation")
    report = {}
    try:
        pairs = _pairs() if (args.codebook or args.golden) else None
        if args.codebook:
            report["codebook"] = materialize_codebook(pairs)
        if args.golden:
            report["golden"] = materialize_golden(pairs)
        if args.evaluation:
            report["evaluation"] = materialize_evaluation(args.replay_dir)
    except TextMismatchError as exc:
        raise SystemExit(f"VERIFICATION FAILED: {exc}")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
