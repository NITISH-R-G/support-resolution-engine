"""Reading and writing the golden set, with the refusals that keep it honest.

Two files, deliberately separate:

* ``candidates.jsonl`` — written once by the sampler, never again. Contains no label field and
  no reply, so nothing that touches it can contaminate an annotation.
* ``annotations.jsonl`` — **append-only**. A re-labelling pass adds a line; it never rewrites
  one. Disagreement between passes is the measurement, so a log that overwrote its own history
  would delete the evidence it exists to produce.

The most important function here fails. ``load_gold`` raises on a partially-labelled set
rather than returning what it has, because a harness that quietly scores 37 of 200 examples
reports a number that looks like a result, and the missing 163 do not appear anywhere in the
output. SPEC section 9.3 states the rule; this is where it is enforced.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Sequence
from pathlib import Path

from hiver_support.golden.schema import (
    AnnotatedExample,
    GoldenAnnotation,
    GoldenCandidate,
    GoldenSetError,
    LabelProvenance,
    Retraction,
)


def write_candidates(
    path: Path, candidates: Sequence[GoldenCandidate], *, overwrite: bool = False
) -> str:
    """Write the candidate file and return its content hash.

    Refuses to overwrite by default: re-sampling once annotation has begun would silently
    discard human work, which is the most expensive thing in this project.
    """
    path = Path(path)
    if path.exists() and not overwrite:
        raise GoldenSetError(
            f"{path} already exists. Re-sampling would discard annotations already collected "
            f"against the current set. Pass overwrite=True only if you mean to start over."
        )
    seen: set[str] = set()
    for candidate in candidates:
        if candidate.pair_id in seen:
            raise GoldenSetError(f"duplicate candidate pair_id {candidate.pair_id!r}")
        seen.add(candidate.pair_id)

    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        json.dumps(c.to_dict(), ensure_ascii=False)
        for c in sorted(candidates, key=lambda c: c.pair_id)
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_candidates(path: Path) -> tuple[GoldenCandidate, ...]:
    path = Path(path)
    if not path.exists():
        raise GoldenSetError(
            f"candidate file not found: {path}. Build it with "
            f"`python scripts/build_golden_candidates.py`."
        )
    return tuple(
        GoldenCandidate.from_dict(payload) for payload in _read_jsonl(path, "candidate")
    )


def append_annotation(path: Path, annotation: GoldenAnnotation) -> None:
    """Append one human labelling action. Never rewrites an existing line."""
    if not isinstance(annotation, GoldenAnnotation):
        raise GoldenSetError(
            f"only a GoldenAnnotation may be appended, got {type(annotation).__name__}"
        )
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(annotation.to_dict(), ensure_ascii=False) + "\n")


def read_annotations(path: Path) -> tuple[GoldenAnnotation, ...]:
    """Read the annotation log. An absent log is an unstarted effort, not an error."""
    path = Path(path)
    if not path.exists():
        return ()
    annotations = []
    for index, payload in enumerate(_read_jsonl(path, "annotation"), start=1):
        if payload.get("record_type") == "retraction":
            # A retraction describes an annotation; it is not one, and it carries no label.
            continue
        declared = payload.get("provenance")
        if declared != LabelProvenance.HUMAN_LABELED.value:
            raise GoldenSetError(
                f"{path} line {index} declares provenance {declared!r}; only HUMAN_LABELED "
                f"records belong in the annotation log"
            )
        annotations.append(GoldenAnnotation.from_dict(payload))
    return tuple(annotations)


def append_retraction(path: Path, retraction: Retraction) -> None:
    """Append a record withdrawing an earlier annotation. Destroys nothing.

    The retracted annotation stays in the log exactly as written, so "this was labelled badly
    and withdrawn" remains a fact anyone can check rather than a claim in a commit message.
    """
    if not isinstance(retraction, Retraction):
        raise GoldenSetError(
            f"only a Retraction may be appended as one, got {type(retraction).__name__}"
        )
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(retraction.to_dict(), ensure_ascii=False) + "\n")


def read_retractions(path: Path) -> tuple[Retraction, ...]:
    """Read the retraction records from the annotation log."""
    path = Path(path)
    if not path.exists():
        return ()
    return tuple(
        Retraction.from_dict(payload)
        for payload in _read_jsonl(path, "annotation")
        if payload.get("record_type") == "retraction"
    )


def effective_annotations(
    annotations: Iterable[GoldenAnnotation], retractions: Iterable[Retraction]
) -> tuple[GoldenAnnotation, ...]:
    """Annotations that still stand, with retracted ones removed.

    A retraction invalidates records for the same example **and pass** written at or before
    its own timestamp. Bounding it in time is what lets the example be annotated properly
    afterwards: an unbounded retraction would silently swallow the replacement too.
    """
    cutoffs: dict[tuple[str, int], str] = {}
    for retraction in retractions:
        key = (retraction.pair_id, retraction.pass_number)
        current = cutoffs.get(key)
        if current is None or retraction.timestamp_utc > current:
            cutoffs[key] = retraction.timestamp_utc

    return tuple(
        annotation
        for annotation in annotations
        if annotation.timestamp_utc
        > cutoffs.get((annotation.pair_id, annotation.pass_number), "")
    )


def load_effective_annotations(path: Path) -> tuple[GoldenAnnotation, ...]:
    """The annotations that count: everything written, minus everything withdrawn."""
    return effective_annotations(read_annotations(path), read_retractions(path))


def _read_jsonl(path: Path, kind: str) -> list[dict]:
    payloads = []
    for index, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            payloads.append(json.loads(line))
        except json.JSONDecodeError as exc:
            # Skipping a corrupt line would silently shrink the evaluation set.
            raise GoldenSetError(f"{path} line {index} is not valid {kind} JSON: {exc}") from exc
    return payloads


def latest_by_pair(
    annotations: Iterable[GoldenAnnotation], pass_number: int | None = None
) -> dict[str, GoldenAnnotation]:
    """Latest annotation per example, optionally restricted to one pass.

    Restricting to a pass is what makes intra-annotator agreement computable: pass 1 and
    pass 2 are compared against each other, not collapsed.
    """
    latest: dict[str, GoldenAnnotation] = {}
    for annotation in annotations:
        if pass_number is not None and annotation.pass_number != pass_number:
            continue
        current = latest.get(annotation.pair_id)
        if current is None or (annotation.pass_number, annotation.timestamp_utc) >= (
            current.pass_number,
            current.timestamp_utc,
        ):
            latest[annotation.pair_id] = annotation
    return latest


def merge(
    candidates: Sequence[GoldenCandidate], annotations: Iterable[GoldenAnnotation]
) -> tuple[AnnotatedExample, ...]:
    """Join candidates to their latest annotations, keeping unlabelled ones visible."""
    known = {c.pair_id for c in candidates}
    latest = latest_by_pair(annotations)
    unknown = sorted(set(latest) - known)
    if unknown:
        raise GoldenSetError(
            f"annotations reference unknown candidates {unknown[:5]}; an annotation with no "
            f"candidate means the set was re-sampled underneath the annotator"
        )
    return tuple(
        AnnotatedExample(candidate=c, annotation=latest.get(c.pair_id))
        for c in sorted(candidates, key=lambda c: c.pair_id)
    )


def coverage(examples: Sequence[AnnotatedExample]) -> dict:
    """Annotation progress, stated in the four provenance classes.

    ``weakly_labelled`` and ``model_generated`` are always zero and are reported anyway: a
    reader should be able to see that they are zero rather than infer it from their absence.
    """
    human = sum(e.is_labelled for e in examples)
    return {
        "total": len(examples),
        "human_labelled": human,
        "unlabelled": len(examples) - human,
        "weakly_labelled": 0,
        "model_generated": 0,
        "complete": bool(examples) and human == len(examples),
        "unlabelled_pair_ids": [e.candidate.pair_id for e in examples if not e.is_labelled],
    }


def load_gold(
    candidates_path: Path, annotations_path: Path, *, require_complete: bool = True
) -> tuple[AnnotatedExample, ...]:
    """Load the golden set for evaluation.

    Raises:
        GoldenSetError: when any example lacks a human label and ``require_complete`` is set.
            This is the refusal SPEC section 9.3 requires — the harness cannot be talked into
            scoring an unlabelled column, and there is no code path that fills one in.
    """
    # Effective, not raw: a retracted label must never reach gold, which is the whole
    # point of being able to retract one.
    examples = merge(
        read_candidates(candidates_path), load_effective_annotations(annotations_path)
    )
    stats = coverage(examples)
    if require_complete and not stats["complete"]:
        missing = stats["unlabelled_pair_ids"]
        raise GoldenSetError(
            f"golden set is not fully annotated: {stats['human_labelled']} of {stats['total']} "
            f"examples are human-labelled. Missing: {missing[:10]}"
            f"{' ...' if len(missing) > 10 else ''}. "
            f"Labels are never backfilled from weak rules, the classifier or an LLM — "
            f"annotate with `python scripts/annotate_golden.py` or evaluate the labelled "
            f"subset explicitly and report its size."
        )
    return examples
