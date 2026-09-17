"""Validating and freezing the golden set, and keeping it evaluation-only.

A golden set is worth exactly what its provenance is worth, so the freeze is a gate rather
than a formality:

**It cannot be frozen early.** A partially annotated set is not gold. Stamping a hash onto
something still changing produces a number that looks authoritative and means nothing.

**Every label is re-checked, not trusted.** The dataclasses validate on construction, but the
freeze reads a file that a person could have edited. Hashing contaminated labels certifies the
contamination, so provenance, taxonomy binding and completeness are all verified again here.

**The hash covers the labels, not just the questions.** A hash over the customer messages
alone would stay identical while every label changed underneath it.

Training code is kept away structurally rather than by convention:
``tests/test_golden_lock.py`` asserts that nothing under ``classifier/``, ``data/``,
``agent/`` or any ``scripts/train_*.py`` imports this package.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from hiver_support.golden.schema import GoldenAnnotation, GoldenCandidate
from hiver_support.taxonomy import TAXONOMY

LOCK_VERSION = "1.0.0"

_REQUIRED_FIELDS = (
    "intent",
    "security_sensitive",
    "context_sufficient",
    "should_escalate",
    "expected_resolution_kind",
    "annotator_id",
)


class GoldenLockError(ValueError):
    """Raised when a golden set cannot be validated or frozen honestly."""


@dataclass(frozen=True, slots=True)
class ValidationReport:
    valid: bool
    problems: tuple[str, ...]
    counts: dict
    distributions: dict

    def to_dict(self) -> dict:
        return {
            "valid": self.valid,
            "problems": list(self.problems),
            "counts": self.counts,
            "distributions": self.distributions,
        }


def validate_golden_set(
    candidates: Sequence[GoldenCandidate],
    annotations: Sequence[GoldenAnnotation],
) -> ValidationReport:
    """Check a golden set against every property the freeze depends on.

    Returns a report rather than raising, so an annotator mid-effort can see exactly what is
    outstanding. ``freeze_golden`` is the function that refuses.
    """
    problems: list[str] = []

    known = {c.pair_id for c in candidates}
    if not candidates:
        problems.append("the candidate set is empty")

    duplicates = [pid for pid, n in Counter(c.pair_id for c in candidates).items() if n > 1]
    if duplicates:
        problems.append(f"duplicate candidate pair_ids: {sorted(duplicates)[:5]}")

    # Latest annotation per example, so a re-labelling pass supersedes rather than duplicates.
    latest: dict[str, GoldenAnnotation] = {}
    unknown: list[str] = []
    non_human: list[str] = []
    missing_fields = 0
    taxonomy_mismatches = 0

    for annotation in annotations:
        pair_id = getattr(annotation, "pair_id", None)
        if pair_id not in known:
            unknown.append(str(pair_id))
            continue
        provenance = getattr(annotation, "provenance", None)
        if provenance is None or not getattr(provenance, "is_gold", False):
            non_human.append(str(pair_id))
            continue
        for field in _REQUIRED_FIELDS:
            value = getattr(annotation, field, None)
            if value is None or (isinstance(value, str) and not value.strip()):
                missing_fields += 1
        if (
            getattr(annotation, "taxonomy_hash", None) != TAXONOMY.frozen_hash
            or getattr(annotation, "intent", None) not in TAXONOMY.names
        ):
            taxonomy_mismatches += 1
            continue
        current = latest.get(pair_id)
        if current is None or annotation.pass_number >= current.pass_number:
            latest[pair_id] = annotation

    if unknown:
        problems.append(
            f"annotations reference unknown candidates: {sorted(set(unknown))[:5]} - the "
            f"candidate set may have been re-sampled underneath the annotator"
        )
    if non_human:
        problems.append(
            f"annotations with non-human provenance: {sorted(set(non_human))[:5]}; only "
            f"HUMAN_LABELED records may enter the golden set"
        )
    if missing_fields:
        problems.append(f"{missing_fields} required field(s) missing across annotations")
    if taxonomy_mismatches:
        problems.append(
            f"{taxonomy_mismatches} annotation(s) do not bind frozen taxonomy "
            f"{TAXONOMY.version}"
        )

    unlabelled = sorted(known - set(latest))
    if unlabelled:
        problems.append(
            f"{len(unlabelled)} of {len(known)} examples are unlabelled: {unlabelled[:10]}"
            f"{' ...' if len(unlabelled) > 10 else ''}"
        )

    labelled = list(latest.values())
    counts = {
        "candidates": len(candidates),
        "human_labelled": len(labelled),
        "unlabelled": len(unlabelled),
        "required_fields_missing": missing_fields,
        "taxonomy_mismatches": taxonomy_mismatches,
        "unknown_candidate_annotations": len(set(unknown)),
        "non_human_annotations": len(set(non_human)),
        "ambiguous": sum(1 for a in labelled if a.is_ambiguous),
        "passes": sorted({a.pass_number for a in labelled}),
        # Anchoring bias is only measurable if the blind subset is countable.
        "blind_entered": sum(
            1 for a in labelled if a.review_action.value == "entered"
        ),
        "model_assisted": sum(
            1 for a in labelled if a.review_action.value != "entered"
        ),
        "suggestions_accepted": sum(
            1 for a in labelled if a.review_action.value == "accepted"
        ),
        "suggestions_corrected": sum(
            1 for a in labelled if a.review_action.value == "corrected"
        ),
    }
    distributions = {
        "intent": dict(Counter(a.intent for a in labelled).most_common()),
        "security_sensitive": dict(Counter(str(a.security_sensitive) for a in labelled)),
        "context_sufficient": dict(Counter(str(a.context_sufficient) for a in labelled)),
        "should_escalate": dict(Counter(str(a.should_escalate) for a in labelled)),
        "expected_resolution_kind": dict(
            Counter(a.expected_resolution_kind.value for a in labelled).most_common()
        ),
        "label_confidence": dict(Counter(a.label_confidence.value for a in labelled)),
        "review_action": dict(Counter(a.review_action.value for a in labelled)),
    }
    return ValidationReport(not problems, tuple(problems), counts, distributions)


def golden_content_hash(
    candidates: Sequence[GoldenCandidate], annotations: Sequence[GoldenAnnotation]
) -> str:
    """Hash the questions **and** their labels.

    Order-independent by construction: a hash that changed when the file was re-sorted would
    make the lock unfalsifiable in practice, because every legitimate rewrite would break it.
    """
    latest: dict[str, GoldenAnnotation] = {}
    for annotation in annotations:
        current = latest.get(annotation.pair_id)
        if current is None or annotation.pass_number >= current.pass_number:
            latest[annotation.pair_id] = annotation

    digest = hashlib.sha256()
    for candidate in sorted(candidates, key=lambda c: c.pair_id):
        annotation = latest.get(candidate.pair_id)
        record = {
            "pair_id": candidate.pair_id,
            "customer_message": candidate.customer_message,
            "intent": getattr(annotation, "intent", None),
            "security_sensitive": getattr(annotation, "security_sensitive", None),
            "context_sufficient": getattr(annotation, "context_sufficient", None),
            "should_escalate": getattr(annotation, "should_escalate", None),
            "expected_resolution_kind": (
                annotation.expected_resolution_kind.value if annotation else None
            ),
            "is_ambiguous": getattr(annotation, "is_ambiguous", None),
            "alternative_intent": getattr(annotation, "alternative_intent", None),
        }
        digest.update(json.dumps(record, sort_keys=True, ensure_ascii=False).encode("utf-8"))
    return digest.hexdigest()


def golden_labels_hash(
    candidates: Sequence[GoldenCandidate], annotations: Sequence[GoldenAnnotation]
) -> str:
    """Hash the labels alone, keyed by pair id: equal for two versions of the set that differ
    only in text (for example v1 and v2, which differ in PII masking)."""
    latest: dict[str, GoldenAnnotation] = {}
    for annotation in annotations:
        current = latest.get(annotation.pair_id)
        if current is None or annotation.pass_number >= current.pass_number:
            latest[annotation.pair_id] = annotation
    digest = hashlib.sha256()
    for candidate in sorted(candidates, key=lambda c: c.pair_id):
        annotation = latest.get(candidate.pair_id)
        record = {
            "pair_id": candidate.pair_id,
            "intent": getattr(annotation, "intent", None),
            "security_sensitive": getattr(annotation, "security_sensitive", None),
            "context_sufficient": getattr(annotation, "context_sufficient", None),
            "should_escalate": getattr(annotation, "should_escalate", None),
            "expected_resolution_kind": (
                annotation.expected_resolution_kind.value if annotation else None
            ),
            "is_ambiguous": getattr(annotation, "is_ambiguous", None),
            "alternative_intent": getattr(annotation, "alternative_intent", None),
        }
        digest.update(json.dumps(record, sort_keys=True, ensure_ascii=False).encode("utf-8"))
    return digest.hexdigest()


def _git_sha() -> str:
    try:
        return (
            subprocess.run(
                ["git", "rev-parse", "--short", "HEAD"],
                cwd=Path(__file__).resolve().parents[3],
                capture_output=True,
                text=True,
            ).stdout.strip()
            or "unknown"
        )
    except Exception:  # pragma: no cover
        return "unknown"


def freeze_golden(
    candidates: Sequence[GoldenCandidate],
    annotations: Sequence[GoldenAnnotation],
) -> dict:
    """Validate, then produce the lock record. Refuses anything short of complete.

    Raises:
        GoldenLockError: if validation fails. A hash over a partial or contaminated set would
            certify exactly the thing it is meant to rule out.
    """
    report = validate_golden_set(candidates, annotations)
    if not report.valid:
        raise GoldenLockError(
            "golden set is not valid and will not be frozen:\n  - "
            + "\n  - ".join(report.problems)
        )

    labelled = {a.pair_id: a for a in annotations if a.pair_id in {c.pair_id for c in candidates}}
    durations = [a.seconds_spent for a in labelled.values() if a.seconds_spent]
    return {
        "lock_version": LOCK_VERSION,
        "usage": "EVALUATION_ONLY",
        "may_be_used_for_training": False,
        "may_be_used_for_threshold_fitting": False,
        "assisted_annotation": any(a.review_action.value != "entered" for a in labelled.values()),
        "_warning": (
            "This is the human-adjudicated GOLDEN SET. It is evaluation-only: no model may be "
            "trained on it and no threshold fitted on it. Labels marked 'accepted' or "
            "'corrected' were produced by MODEL-ASSISTED PRE-ANNOTATION that a human reviewed "
            "and adopted or changed; labels marked 'entered' were written blind with no "
            "suggestion shown. These are NOT 200 independently-classified-from-scratch "
            "labels, and must never be described as such. The blind subset is what makes "
            "anchoring bias measurable - see counts.blind_entered."
        ),
        "frozen_at": datetime.now(timezone.utc).isoformat(),
        "git_sha": _git_sha(),
        "taxonomy_version": TAXONOMY.version,
        "taxonomy_hash": TAXONOMY.frozen_hash,
        "content_sha256": golden_content_hash(candidates, annotations),
        "annotators": sorted({a.annotator_id for a in labelled.values()}),
        "annotation_versions": sorted({a.annotation_version for a in labelled.values()}),
        "counts": report.counts,
        "distributions": report.distributions,
        "annotation_seconds_total": round(sum(durations), 1) if durations else None,
        "annotation_seconds_median": (
            round(sorted(durations)[len(durations) // 2], 1) if durations else None
        ),
        "protocol": "docs/GOLDEN_SET.md",
    }


def verify_lock(
    lock: dict,
    candidates: Sequence[GoldenCandidate],
    annotations: Sequence[GoldenAnnotation],
) -> bool:
    """Confirm a set still matches its lock. Raises on any drift."""
    actual = golden_content_hash(candidates, annotations)
    if actual != lock.get("content_sha256"):
        raise GoldenLockError(
            f"golden set does not match its lock: content hash {actual[:16]}... != "
            f"{str(lock.get('content_sha256'))[:16]}... The set changed after freezing, which "
            f"invalidates every metric computed against it."
        )
    if lock.get("taxonomy_hash") != TAXONOMY.frozen_hash:
        raise GoldenLockError("lock was made against a different frozen taxonomy")
    return True
