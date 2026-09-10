"""Measure what support actually DID, so merge/split decisions rest on operational evidence.

A label earns its place by changing the support action, not by having distinct vocabulary
(`SPEC.md` §5). The brand's historical replies are the record of that action, so comparing
reply behaviour between two candidate groups tests whether a proposed distinction is
operational or merely lexical.

Every comparison carries a bootstrap confidence interval. Candidate groups differ wildly in
size — some hypotheses have a few hundred examples against tens of thousands — and a raw rate
gap between a 277-message group and a 16,000-message group is exactly the kind of number that
looks decisive and means nothing. A difference whose interval spans zero is reported as no
difference.

Scope: replies inform taxonomy DESIGN, from the train split only. They must never be used to
assign labels during annotation; see `docs/ANNOTATION_GUIDE.md`.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from hiver_support.data.reply_classify import classify_reply

_QUESTION_RE = re.compile(r"\?")
_SUPPORT_LINK_RE = re.compile(r"\[URL\]|support\.apple\.com|apple\.co/", re.IGNORECASE)

METRICS = (
    "deflection_rate",
    "actionable_rate",
    "substantive_rate",
    "asks_question_rate",
    "links_to_support_rate",
)

BOOTSTRAP_ITERATIONS = 2_000
CONFIDENCE = 0.95

# Below this, no verdict is issued regardless of the apparent gap. The bootstrap degenerates on
# tiny samples: resampling two identical values reproduces them every time, so a 2-vs-2
# comparison yields a confidence interval of [1, 1] and declares a difference from almost no
# evidence. Candidate groups here range from a few hundred to tens of thousands, so guarding
# this is what stops a rare category's noise from being read as an operational distinction.
MIN_GROUP_FOR_VERDICT = 30


@dataclass(frozen=True, slots=True)
class HandlingProfile:
    """How support handled one candidate group. ``n`` is exposed so rates are never quoted alone."""

    label: str
    n: int
    deflection_rate: float
    actionable_rate: float
    substantive_rate: float
    asks_question_rate: float
    links_to_support_rate: float
    mean_reply_words: float

    def as_dict(self) -> dict:
        return {
            "label": self.label,
            "n": self.n,
            **{metric: getattr(self, metric) for metric in METRICS},
            "mean_reply_words": self.mean_reply_words,
        }


@dataclass(frozen=True, slots=True)
class ProfileDifference:
    """A between-group difference with its uncertainty. Never report ``difference`` alone."""

    metric: str
    a: float
    b: float
    difference: float
    ci_low: float
    ci_high: float
    significant: bool


def _metric_vectors(replies: Sequence[str]) -> dict[str, np.ndarray]:
    classifications = [classify_reply(reply) for reply in replies]
    return {
        "deflection_rate": np.array([c.is_deflection for c in classifications], dtype=float),
        "actionable_rate": np.array([c.is_actionable for c in classifications], dtype=float),
        "substantive_rate": np.array([c.is_substantive for c in classifications], dtype=float),
        "asks_question_rate": np.array(
            [bool(_QUESTION_RE.search(r)) for r in replies], dtype=float
        ),
        "links_to_support_rate": np.array(
            [bool(_SUPPORT_LINK_RE.search(r)) for r in replies], dtype=float
        ),
    }


def handling_profile(label: str, replies: Sequence[str]) -> HandlingProfile:
    """Summarise support behaviour over one group of replies."""
    if not replies:
        return HandlingProfile(label, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    vectors = _metric_vectors(replies)
    return HandlingProfile(
        label=label,
        n=len(replies),
        deflection_rate=round(float(vectors["deflection_rate"].mean()), 4),
        actionable_rate=round(float(vectors["actionable_rate"].mean()), 4),
        substantive_rate=round(float(vectors["substantive_rate"].mean()), 4),
        asks_question_rate=round(float(vectors["asks_question_rate"].mean()), 4),
        links_to_support_rate=round(float(vectors["links_to_support_rate"].mean()), 4),
        mean_reply_words=round(float(np.mean([len(r.split()) for r in replies])), 2),
    )


def compare_handling(
    a_replies: Sequence[str],
    b_replies: Sequence[str],
    *,
    seed: int = 20260910,
    iterations: int = BOOTSTRAP_ITERATIONS,
) -> list[ProfileDifference]:
    """Compare two groups metric by metric, with a bootstrap CI on each difference.

    A group smaller than ``MIN_GROUP_FOR_VERDICT`` yields no verdict: every difference is
    returned as non-significant with an interval spanning the full range, because a merge or
    split decision cannot be supported by data that thin.
    """
    if len(a_replies) < MIN_GROUP_FOR_VERDICT or len(b_replies) < MIN_GROUP_FOR_VERDICT:
        return [
            ProfileDifference(metric, 0.0, 0.0, 0.0, -1.0, 1.0, False) for metric in METRICS
        ]

    a_vectors = _metric_vectors(a_replies)
    b_vectors = _metric_vectors(b_replies)
    rng = np.random.default_rng(seed)
    low_q, high_q = (1 - CONFIDENCE) / 2 * 100, (1 + CONFIDENCE) / 2 * 100

    results = []
    for metric in METRICS:
        a_vector, b_vector = a_vectors[metric], b_vectors[metric]
        observed = float(a_vector.mean() - b_vector.mean())

        a_samples = rng.choice(a_vector, size=(iterations, len(a_vector)), replace=True).mean(1)
        b_samples = rng.choice(b_vector, size=(iterations, len(b_vector)), replace=True).mean(1)
        differences = a_samples - b_samples

        ci_low = float(np.percentile(differences, low_q))
        ci_high = float(np.percentile(differences, high_q))
        results.append(
            ProfileDifference(
                metric=metric,
                a=round(float(a_vector.mean()), 4),
                b=round(float(b_vector.mean()), 4),
                difference=round(observed, 4),
                ci_low=round(ci_low, 4),
                ci_high=round(ci_high, 4),
                significant=bool(ci_low > 0 or ci_high < 0),
            )
        )
    return results
