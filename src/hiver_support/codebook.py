"""Codebook example texts, kept out of the repository and rebuilt locally from the dataset.

The taxonomy's 29 codebook examples are real customer messages. Their text is not committed:
``codebook_examples.json`` stores, per example, the source pair id, an edit script against that
tweet's raw text, and the sha256 of the result. ``scripts/materialize_text.py`` rebuilds the
texts from the Kaggle download into ``data/local/codebook_examples.json`` (gitignored), and
checks every hash.

The texts matter because taxonomy v0.3.0's frozen hash covers them, and all 200 golden
annotations bind that hash. With the texts materialised, ``Taxonomy.frozen_hash`` recomputes
the hash and raises on any mismatch. Without them it returns the recorded value, which is what
annotations and locks were written against.
"""
from __future__ import annotations

import hashlib
import json
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EDIT_SCRIPTS = Path(__file__).resolve().parent / "codebook_examples.json"
MATERIALIZED = ROOT / "data" / "local" / "codebook_examples.json"


class CodebookError(ValueError):
    """Raised when a rebuilt codebook text does not match its recorded hash."""


@lru_cache(maxsize=1)
def edit_scripts() -> dict:
    return json.loads(EDIT_SCRIPTS.read_text(encoding="utf-8"))


def frozen_taxonomy_hash() -> str:
    return edit_scripts()["taxonomy_frozen_hash"]


@lru_cache(maxsize=1)
def _materialized() -> dict[str, str]:
    if not MATERIALIZED.exists():
        return {}
    return json.loads(MATERIALIZED.read_text(encoding="utf-8"))["texts"]


def is_materialized() -> bool:
    return bool(_materialized())


def example_text(pair_id: str) -> str:
    """The rebuilt text, or "" when the codebook has not been materialised on this machine."""
    return _materialized().get(pair_id, "")


def apply_edits(source: str, edits: list) -> str:
    out, position = [], 0
    for start, end, inserted in edits:
        out.append(source[position:start])
        out.append(inserted)
        position = end
    out.append(source[position:])
    return "".join(out)


def rebuild(customer_text_by_pair: dict[str, str]) -> dict[str, str]:
    """Rebuild every example text and verify its sha256. Raises on the first mismatch."""
    texts = {}
    for example in edit_scripts()["examples"]:
        source = customer_text_by_pair.get(example["pair_id"])
        if source is None:
            raise CodebookError(f"source pair {example['pair_id']} not found in the local corpus")
        text = apply_edits(source, example["edits"])
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        if digest != example["sha256"]:
            raise CodebookError(f"codebook example {example['pair_id']} does not match its recorded hash")
        texts[example["pair_id"]] = text
    return texts
