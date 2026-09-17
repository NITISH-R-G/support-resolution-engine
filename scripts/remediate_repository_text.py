"""ONE-OFF (release remediation): remove tweet text from the repository without losing anything.

Run once, on the tree that still contains text, with the Kaggle download present. Every step is
verified before anything is overwritten, and the full-text originals are kept under data/local/.

1. Record the sha256 of each original full-text artifact (LF-normalised bytes) in
   ``reports/golden_eval/ORIGINAL_ARTIFACT_HASHES.json``. Materialisation must reproduce these.
2. Golden candidates:
   - rebuild v1 from the dataset with the frozen v1 masker; it must equal the committed file
     record for record;
   - build v2 with the corrected masker;
   - write the text-free record file, both local full-text versions and ``INTEGRITY.json``.
3. Evaluation artifacts: replace with text-free versions (``publish_eval_artifacts``).
4. Every other tracked file: redact matched tweet text with the shared detector
   (``scripts/audit/tweet_text.py``), except files handled structurally and ``taxonomy.py``
   (codebook edit scripts).

Usage:
    python scripts/remediate_repository_text.py            # dry run: report, write nothing
    python scripts/remediate_repository_text.py --apply
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "scripts" / "audit"))

from discover_taxonomy import load_brand_pairs  # noqa: E402
from publish_eval_artifacts import publish  # noqa: E402
from tweet_text import MARKER_RE, TweetIndex, scan_bytes  # noqa: E402

from hiver_support.data.pii import mask_pii, mask_pii_v1  # noqa: E402
from hiver_support.data.split import temporal_split  # noqa: E402
from hiver_support.golden import paths  # noqa: E402
from hiver_support.golden.sampling import _to_candidate, context_turns  # noqa: E402
from hiver_support.golden.textfree import candidate_record  # noqa: E402

EVAL = ROOT / "reports" / "golden_eval"
EVAL_FILES = ("predictions.jsonl", "judge.jsonl", "judge_claude_partial_402.jsonl")
STRUCTURAL = {
    "data/golden/candidates.jsonl",
    *(f"reports/golden_eval/{name}" for name in EVAL_FILES),
    "src/hiver_support/taxonomy.py",
    # Files that implement or document the detector itself carry no tweet text.
    "scripts/audit/tweet_text.py",
}


def lf_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def golden(apply: bool) -> dict:
    committed_lines = [l for l in paths.CANDIDATE_RECORDS.read_bytes().replace(b"\r\n", b"\n").decode("utf-8").splitlines() if l.strip()]
    committed = [json.loads(l) for l in committed_lines]
    pool = {p.pair_id: p for p in temporal_split(load_brand_pairs(None)).test_pool}

    v1, v2, records = [], [], []
    for line, row in zip(committed_lines, committed):
        pair = pool[row["pair_id"]]
        rebuilt_v1 = _to_candidate(pair, masker=mask_pii_v1)
        if json.dumps(rebuilt_v1.to_dict(), ensure_ascii=False) != line:
            raise SystemExit(f"v1 rebuild differs from the committed record for {row['pair_id']}; aborting")
        rebuilt_v2 = _to_candidate(pair, masker=mask_pii)
        v1.append(rebuilt_v1)
        v2.append(rebuilt_v2)
        records.append(candidate_record(rebuilt_v1.to_dict(), rebuilt_v2.to_dict(),
                                        [t.tweet_id for t in context_turns(pair)]))

    changed = [a.pair_id for a, b in zip(v1, v2) if a.to_dict() != b.to_dict()]
    # v2 differs from v1 only by applying the corrected masker to v1 text: PII placeholders, nothing else.
    for a, b in zip(v1, v2):
        if mask_pii(a.customer_message).text != b.customer_message:
            raise SystemExit(f"{a.pair_id}: v2 is not v1 with only PII re-masked")
        for ta, tb in zip(a.context, b.context):
            if mask_pii(ta.text).text != tb.text:
                raise SystemExit(f"{a.pair_id}: v2 context is not v1 context re-masked")

    from hiver_support.golden.store import write_candidates

    result = {"records": len(records), "v1_rebuilt_identical": len(v1), "v2_records_changed": changed}
    if apply:
        hashes = {}
        for version, candidates in (("v1", v1), ("v2", v2)):
            target = paths.local_candidates(version)
            hashes[version] = write_candidates(target, candidates, overwrite=True)
        manifest = json.loads((paths.GOLDEN_DIR / "manifest.json").read_text(encoding="utf-8"))
        integrity = {
            "_doc": (
                "Platform-independent integrity for the golden set. The committed candidates.jsonl "
                "is text-free; full text is rebuilt locally by scripts/materialize_text.py. File "
                "hashes here are over LF bytes. manifest.json's candidates_sha256 was computed on a "
                "Windows CRLF working copy at sampling time and is kept unchanged as history."
            ),
            "v1": {
                "candidates_lf_sha256": hashes["v1"],
                "lock": "data/golden/GOLDEN_LOCK.json",
                "masker": "mask_pii_v1",
                "manifest_candidates_sha256_windows_crlf": manifest["candidates_sha256"],
            },
            "v2": {
                "candidates_lf_sha256": hashes["v2"],
                "lock": "data/golden/GOLDEN_LOCK_V2.json",
                "masker": "mask_pii",
                "records_differing_from_v1": changed,
            },
        }
        paths.INTEGRITY.write_bytes((json.dumps(integrity, indent=2) + "\n").encode("utf-8"))
        paths.CANDIDATE_RECORDS.write_bytes(
            ("\n".join(json.dumps(r, ensure_ascii=False) for r in records) + "\n").encode("utf-8"))
        result["local_hashes"] = hashes
    return result


def evaluation(apply: bool) -> dict:
    originals = {name: lf_sha256(EVAL / name) for name in EVAL_FILES}
    originals["data/golden/candidates.jsonl"] = lf_sha256(paths.CANDIDATE_RECORDS)
    if apply:
        backup = paths.LOCAL_DIR / "golden_eval_original"
        backup.mkdir(parents=True, exist_ok=True)
        for name in EVAL_FILES:
            (backup / name).write_bytes((EVAL / name).read_bytes().replace(b"\r\n", b"\n"))
        (EVAL / "ORIGINAL_ARTIFACT_HASHES.json").write_bytes((json.dumps({
            "_doc": ("sha256 of the original full-text evaluation artifacts (LF bytes), recorded "
                     "before redaction. scripts/materialize_text.py --evaluation must rebuild files "
                     "with exactly these hashes."),
            "sha256": originals,
        }, indent=2) + "\n").encode("utf-8"))
        publish(backup, EVAL)
    return {"original_hashes": originals}


# Keys whose values are tweet text or model output derived from a customer message, in the
# exploratory report JSON. Hashed whatever the detector says: a model reply is not verbatim tweet
# text, but it is derived from one. In the smoke reports only, ``probes`` sections hold synthetic
# probe sentences written for this project and are kept (taxonomy_probes.json's probes are real).
DERIVED_TEXT_KEYS = {"message", "reply", "reason_detail", "customer", "brand_reply", "text",
                     "rationale", "customer_text", "resolution_text", "draft", "response"}
QUOTED_EXAMPLE = re.compile(r'\*"([^"*]+)"\*')
# Idempotence: a value already replaced by either marker form is left alone.
ANY_MARKER = re.compile(r"\[(tweet-)?text redacted: [^\]]*\]")


def text_marker(value: str) -> str:
    return f"[text redacted: sha256={hashlib.sha256(value.encode('utf-8')).hexdigest()[:16]}]"


def redact_report_keys(value, inside_probes: bool = False, probes_are_synthetic: bool = False):
    count = 0
    if isinstance(value, dict):
        out = {}
        for key, item in value.items():
            if (isinstance(item, str) and key in DERIVED_TEXT_KEYS and not inside_probes and item.strip()
                    and not MARKER_RE.fullmatch(item) and not ANY_MARKER.fullmatch(item)):
                out[key] = text_marker(item)
                count += 1
            else:
                out[key], n = redact_report_keys(
                    item, inside_probes or (probes_are_synthetic and key == "probes"), probes_are_synthetic)
                count += n
        return out, count
    if isinstance(value, list):
        items = [redact_report_keys(item, inside_probes, probes_are_synthetic) for item in value]
        return [i for i, _ in items], sum(n for _, n in items)
    return value, 0


def redact_quoted_examples(text: str) -> tuple[str, int]:
    """failure_analysis.md quotes messages and agent replies in *"..."* spans; hash them all."""
    count = len(QUOTED_EXAMPLE.findall(text))
    return QUOTED_EXAMPLE.sub(lambda m: "*" + text_marker(m.group(1)) + "*", text), count


def everything_else(apply: bool) -> dict:
    index = TweetIndex()
    tracked = subprocess.run(["git", "ls-files"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.split("\n")
    report = {}
    for rel in filter(None, tracked):
        if rel in STRUCTURAL or not (ROOT / rel).is_file():
            continue
        data = (ROOT / rel).read_bytes()
        matches, redacted = scan_bytes(rel, data, index, redact=True)
        extra = 0
        if rel.startswith("reports/") and rel.endswith(".json"):
            document, extra = redact_report_keys(json.loads(redacted.decode("utf-8")),
                                                 probes_are_synthetic=rel.startswith("reports/llm_smoke"))
            if extra:
                indent = 2 if b"\n  " in data[:200] else None
                redacted = (json.dumps(document, indent=indent, ensure_ascii=False) + "\n").encode("utf-8")
        if rel == "reports/golden_eval/failure_analysis.md":
            text, extra = redact_quoted_examples(redacted.decode("utf-8"))
            redacted = text.encode("utf-8")
        if matches or extra:
            report[rel] = {"tweet_text_matches": len(matches), "derived_text_fields": extra}
            if apply:
                (ROOT / rel).write_bytes(redacted.replace(b"\r\n", b"\n"))
    return report


def suggestion_rationales(apply: bool) -> dict:
    """Pre-annotator rationales: model output written about customer messages.

    Hashed in place in suggestions.jsonl and in each annotation's embedded model_suggestion.
    Labels are untouched, so both golden locks still verify. The original file hashes are
    recorded, and materialize_text.py --golden rebuilds both files from the pre-annotator's
    cached responses and checks them.
    """
    from hiver_support.golden.textfree import sha256_text

    files = {"suggestions": paths.GOLDEN_DIR / "suggestions.jsonl", "annotations": paths.ANNOTATIONS}
    originals = {name: lf_sha256(path) for name, path in files.items()}
    rewritten, count = {}, 0
    for name, path in files.items():
        rows = []
        for line in path.read_bytes().replace(b"\r\n", b"\n").decode("utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            target = row if name == "suggestions" else row.get("model_suggestion")
            if isinstance(target, dict) and "rationale" in target:
                replaced = {("rationale_sha256" if k == "rationale" else k):
                            (sha256_text(v) if k == "rationale" else v) for k, v in target.items()}
                if name == "suggestions":
                    row = replaced
                else:
                    row = {k: (replaced if k == "model_suggestion" else v) for k, v in row.items()}
                count += 1
            rows.append(row)
        rewritten[name] = ("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n").encode("utf-8")
    if apply:
        integrity = json.loads(paths.INTEGRITY.read_text(encoding="utf-8"))
        integrity["v1"]["suggestions_original_lf_sha256"] = originals["suggestions"]
        integrity["v1"]["annotations_original_lf_sha256"] = originals["annotations"]
        integrity["v1"]["rationales"] = (
            "Pre-annotator rationales are stored as rationale_sha256 in suggestions.jsonl and in each "
            "annotation's model_suggestion; materialize_text.py --golden rebuilds both files from the "
            "local pre-annotator response cache and checks these original hashes.")
        paths.INTEGRITY.write_bytes((json.dumps(integrity, indent=2) + "\n").encode("utf-8"))
        for name, path in files.items():
            path.write_bytes(rewritten[name])
    return {"rationales_hashed": count, "original_lf_sha256": originals}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--only-suggestions", action="store_true",
                        help="second pass: pre-annotator rationales only")
    args = parser.parse_args()
    if args.only_suggestions:
        print(json.dumps(suggestion_rationales(args.apply), indent=2))
        return
    out = {"evaluation": evaluation(args.apply)}  # hashes recorded before candidates are rewritten
    out["golden"] = golden(args.apply)
    out["generic_redaction"] = everything_else(args.apply)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
