"""Where golden-set and evaluation files live, committed (text-free) versus local (full text).

The repository holds no tweet text. Committed files carry ids, labels, metadata and sha256
hashes. ``scripts/materialize_text.py`` rebuilds the full-text files from the local Kaggle
download into ``data/local/`` (gitignored), verifying every hash, and the tools that need text
read them from there.

Golden-set versions:
    v1  the set exactly as annotated and evaluated. Built with the frozen v1 PII masker; bound
        by ``GOLDEN_LOCK.json``. Immutable.
    v2  the same 200 examples and labels with the corrected PII masker (``mask_pii``); bound by
        ``GOLDEN_LOCK_V2.json``. For future evaluations; the reported evaluation used v1.
"""
from __future__ import annotations

from pathlib import Path

from hiver_support.golden.schema import GoldenSetError

ROOT = Path(__file__).resolve().parents[3]
GOLDEN_DIR = ROOT / "data" / "golden"
CANDIDATE_RECORDS = GOLDEN_DIR / "candidates.jsonl"  # committed, text-free
ANNOTATIONS = GOLDEN_DIR / "annotations.jsonl"
LOCK = {"v1": GOLDEN_DIR / "GOLDEN_LOCK.json", "v2": GOLDEN_DIR / "GOLDEN_LOCK_V2.json"}
INTEGRITY = GOLDEN_DIR / "INTEGRITY.json"

LOCAL_DIR = ROOT / "data" / "local"
VERSIONS = ("v1", "v2")


def local_candidates(version: str = "v1") -> Path:
    if version not in VERSIONS:
        raise GoldenSetError(f"unknown golden-set version {version!r}; expected one of {VERSIONS}")
    return LOCAL_DIR / "golden" / version / "candidates.jsonl"


def require_local_candidates(version: str = "v1") -> Path:
    """The materialised full-text candidate file, or a clear error saying how to create it."""
    path = local_candidates(version)
    if not path.exists():
        raise GoldenSetError(
            f"{path} does not exist. The repository stores no tweet text; rebuild it from the "
            f"Kaggle download with: python scripts/materialize_text.py --golden"
        )
    return path


def local_eval_dir() -> Path:
    return LOCAL_DIR / "golden_eval"
