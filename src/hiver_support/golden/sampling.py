"""Deterministic stratified sampling of golden candidates from the held-out test pool.

Implements ``docs/GOLDEN_SET.md`` section 3, which was frozen before this module drew a single
example. Three properties carry the weight:

**The reservoir is drawn first.** Fifty examples are taken uniformly at random from the whole
pool before any stratification happens. If every stratification proxy below turns out to be
wrong about what is hard, those fifty are still a random sample of real traffic and still
support an honest — if wide — estimate of production behaviour.

**Stratification uses the weak labelling functions, and says so.** That is the only place a
weak signal touches the golden set. It decides *which examples are shown*, never *what they
are labelled*: the stratum is written to the sampling frame, not to the file the annotator
reads. Bounded that way it is safe; unsaid it would be contamination.

**Inclusion probabilities are recorded.** The design deliberately over-samples hard strata, so
the raw sample estimates nothing about production. Because every inclusion probability is
known by construction rather than fitted, a representative estimate stays computable — and the
report is obliged to give the reweighted number alongside the flattering per-stratum ones.
"""

from __future__ import annotations

import hashlib
import subprocess
from collections import Counter, OrderedDict
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from hiver_support.classifier.weak_labels import WEAK_PROVENANCE, weak_label
from hiver_support.data.normalise import normalise_text
from hiver_support.data.pii import mask_pii
from hiver_support.data.schema import SupportPair
from hiver_support.golden.schema import (
    ContextTurn,
    GoldenCandidate,
    LabelProvenance,
    assert_candidate_schema_is_blind,
)
from hiver_support.taxonomy import TAXONOMY

BRAND = "AppleSupport"
SEED = 20260911
GOLDEN_SIZE = 200
RESERVOIR_SIZE = 50
LONG_MESSAGE_WORDS = 40

# Priority order, frozen in docs/GOLDEN_SET.md section 3.2. First match wins, so rarest and
# most safety-critical comes first: a broader stratum absorbing the security cases would leave
# the gate that matters most untested.
STRATUM_QUOTAS: OrderedDict[str, int] = OrderedDict(
    security_signal=25,
    thin_context=25,
    multi_signal=20,
    rule_abstain=40,
    long_message=15,
    typical=25,
)
STRATUM_ORDER = tuple(STRATUM_QUOTAS)
FALLBACK_STRATUM = "typical"


class SamplingError(ValueError):
    """Raised when a sample cannot be drawn honestly. Never downgraded to a warning."""


@dataclass(frozen=True, slots=True)
class SamplingRecord:
    """Why one example is in the set, and how much it should count for.

    Lives in the sampling frame, never in the candidate file: ``stratum`` and
    ``weak_label`` are exactly the hints that would anchor an annotator.
    """

    pair_id: str
    component: str
    stratum: str
    inclusion_probability: float
    weight: float
    weak_label: dict

    def to_dict(self) -> dict:
        return {
            "pair_id": self.pair_id,
            "component": self.component,
            "stratum": self.stratum,
            "inclusion_probability": round(self.inclusion_probability, 8),
            "weight": round(self.weight, 6),
            "stratification_signal": self.weak_label,
            "stratification_signal_provenance": LabelProvenance.WEAKLY_LABELED.value,
        }


@dataclass(frozen=True, slots=True)
class SamplingResult:
    candidates: tuple[GoldenCandidate, ...]
    records: tuple[SamplingRecord, ...]
    manifest: dict


def assign_stratum(text: str) -> str:
    """Assign one stratum by the frozen priority order. Pure function of the text.

    Every signal comes from ``weak_labels``; nothing new is invented here, so the stratum
    means exactly what the training-label rules mean and inherits exactly their blind spots.
    """
    if not isinstance(text, str):
        raise TypeError(f"assign_stratum expects str, got {type(text).__name__}")

    label = weak_label(text)
    if label.security_sensitive:
        return "security_signal"
    if not label.context_sufficient:
        return "thin_context"
    if label.has_conflict:
        return "multi_signal"
    if label.abstained:
        return "rule_abstain"
    if len(text.split()) >= LONG_MESSAGE_WORDS:
        return "long_message"
    return FALLBACK_STRATUM


NEAR_DUPLICATE_CHUNK = 512


def _near_duplicate_mask(
    pool_texts: Sequence[str], corpus_texts: Sequence[str], threshold: float
) -> np.ndarray:
    """Which pool texts have a corpus near-duplicate at or above ``threshold``.

    Mirrors ``leakage.assert_no_near_duplicates`` exactly — same analyser, same n-gram range,
    same cosine — so that filtering here genuinely pre-empts the guard rather than
    approximating it. Computed in chunks because the dense similarity matrix for the full
    test pool against a 20,000-pair corpus would be several gigabytes.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer

    mask = np.zeros(len(pool_texts), dtype=bool)
    if not pool_texts or not corpus_texts or not any(corpus_texts) or not any(pool_texts):
        return mask

    vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=1)
    corpus_matrix = vectorizer.fit_transform(corpus_texts)
    pool_matrix = vectorizer.transform(pool_texts)

    for start in range(0, pool_matrix.shape[0], NEAR_DUPLICATE_CHUNK):
        block = pool_matrix[start : start + NEAR_DUPLICATE_CHUNK]
        # Both matrices are L2-normalised by TfidfVectorizer, so the dot product is cosine.
        similarities = (block @ corpus_matrix.T).toarray()
        mask[start : start + block.shape[0]] = similarities.max(axis=1) >= threshold
    return mask


def filter_eligible(
    pool: Sequence[SupportPair],
    corpus: Sequence[SupportPair],
    *,
    near_duplicate_corpus: Sequence[SupportPair] | None = None,
    threshold: float = 0.90,
) -> tuple[list[SupportPair], dict]:
    """Drop test-pool pairs that would violate a leakage guard, before sampling begins.

    Both exclusions below were **found by running the real guards on a real draw**, not
    anticipated:

    * 546 of 22,378 test-pool messages (2.4%) are verbatim duplicates of a train message once
      normalised — overwhelmingly bare acknowledgements like "yes", "ok" and "thanks", which
      every customer sends in the same words.
    * Further messages are near-duplicates at cosine ≥ 0.90, e.g. golden ``7plus ios 11 1 2``
      against corpus ``7plus ios 11 1``. Two device-plus-version fragments differing by one
      character are the same query for retrieval purposes.

    The guards are right and are not negotiable, so eligibility is decided *here*, before any
    draw, rather than by removing inconvenient examples afterwards. Filtering once you can see
    which examples came out is how a sampling frame becomes a way to pick a result.

    **This filter biases the set**, and the bias is reported rather than absorbed: it strips
    a large share of the ``thin_context`` stratum, so the golden set under-represents
    ultra-short messages relative to real traffic. See ``docs/GOLDEN_SET.md`` section 3.6.
    """
    from hiver_support.leakage import normalise_for_comparison

    corpus_texts = {normalise_for_comparison(p.customer_text) for p in corpus}
    corpus_texts.discard("")

    survivors, excluded_exact = [], []
    for pair in pool:
        if normalise_for_comparison(pair.customer_text) in corpus_texts:
            excluded_exact.append(pair)
        else:
            survivors.append(pair)

    excluded_near: list[SupportPair] = []
    if near_duplicate_corpus:
        mask = _near_duplicate_mask(
            [normalise_for_comparison(p.customer_text) for p in survivors],
            [normalise_for_comparison(p.customer_text) for p in near_duplicate_corpus],
            threshold,
        )
        eligible = [p for p, near in zip(survivors, mask) if not near]
        excluded_near = [p for p, near in zip(survivors, mask) if near]
    else:
        eligible = survivors

    excluded = excluded_exact + excluded_near
    report = {
        "rule": (
            "a golden candidate's normalised text must neither appear verbatim in the "
            "retrieval corpus nor be a near-duplicate of one at cosine >= threshold"
        ),
        "guards": ["leakage.assert_no_text_duplicates", "leakage.assert_no_near_duplicates"],
        "applied": "before sampling, to the whole test pool",
        "near_duplicate_threshold": threshold,
        "near_duplicate_corpus_size": len(near_duplicate_corpus or ()),
        "pool_size": len(pool),
        "eligible": len(eligible),
        "excluded": len(excluded),
        "excluded_exact_duplicate": len(excluded_exact),
        "excluded_near_duplicate": len(excluded_near),
        "excluded_rate": round(len(excluded) / len(pool), 4) if pool else 0.0,
        "excluded_by_stratum": dict(
            Counter(assign_stratum(p.customer_text) for p in excluded).most_common()
        ),
        "bias_introduced": (
            "Excluded messages are overwhelmingly short acknowledgements and bare "
            "device/version fragments, so the golden set under-represents ultra-short "
            "messages and therefore under-represents context_sufficient=false relative to "
            "production traffic."
        ),
    }
    return eligible, report


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
    except Exception:  # pragma: no cover - git absent is not a sampling failure
        return "unknown"


def context_turns(pair: SupportPair) -> list:
    """The thread turns shown with a candidate: those strictly before the message, in order."""
    cutoff = pair.customer_tweet.created_at
    return [
        turn
        for turn in sorted(pair.context, key=lambda t: (t.created_at, t.tweet_id))
        if turn.created_at < cutoff and turn.tweet_id != pair.support_tweet.tweet_id
    ]


def _to_candidate(pair: SupportPair, masker=mask_pii) -> GoldenCandidate:
    """Build the blind record an annotator sees.

    The support reply is not read. Context is filtered to turns that genuinely *precede* the
    message, because a later turn is information the agent could not have had and would let
    the annotator label from the outcome.

    ``masker`` is ``mask_pii`` for new candidates. Rebuilding golden-set v1 exactly passes
    ``mask_pii_v1``, the masker v1 was sampled with.
    """
    message = masker(normalise_text(pair.customer_text, brand=BRAND)).text
    context = tuple(
        ContextTurn(
            author_role="customer" if turn.inbound else "brand",
            text=masker(normalise_text(turn.text, brand=BRAND)).text,
            created_at=turn.created_at,
        )
        for turn in context_turns(pair)
    )
    return GoldenCandidate(
        pair_id=pair.pair_id,
        conversation_id=pair.conversation_id,
        customer_tweet_id=pair.customer_tweet.tweet_id,
        customer_message=message,
        created_at=pair.customer_tweet.created_at,
        context=context,
    )


def sample_candidates(
    pairs: Sequence[SupportPair],
    *,
    size: int = GOLDEN_SIZE,
    reservoir: int = RESERVOIR_SIZE,
    seed: int = SEED,
) -> SamplingResult:
    """Draw the golden candidate set from the held-out test pool.

    Args:
        pairs: the test pool, and only the test pool. Train and dev are excluded by being
            handed to a different variable, not by a filter that could be forgotten.
        size: total examples to draw.
        reservoir: how many of those are the unstratified random core.
        seed: frozen in the protocol; a different seed is a different study.

    Raises:
        SamplingError: when the pool is too small, contains duplicate ids, or the requested
            composition is impossible. Returning a short sample would look like a result.
    """
    assert_candidate_schema_is_blind()

    if not pairs:
        raise SamplingError("cannot sample from an empty test pool")
    if reservoir >= size:
        raise SamplingError(f"reservoir {reservoir} must be smaller than size {size}")
    if len(pairs) < size:
        raise SamplingError(
            f"test pool too small: {len(pairs)} pairs for a target of {size}. Report the size "
            f"actually available rather than sampling with replacement."
        )
    duplicates = [pid for pid, n in Counter(p.pair_id for p in pairs).items() if n > 1]
    if duplicates:
        raise SamplingError(
            f"duplicate pair_ids in the test pool: {sorted(duplicates)[:5]}; a duplicated "
            f"example would be annotated twice and counted twice"
        )

    # Sorting first makes the draw independent of how the pool arrived, so a change to the
    # loading order cannot silently change which examples are evaluated.
    ordered = sorted(pairs, key=lambda p: p.pair_id)
    rng = np.random.default_rng(seed)
    population = len(ordered)

    reservoir_idx = sorted(int(i) for i in rng.choice(population, size=reservoir, replace=False))
    reservoir_ids = {ordered[i].pair_id for i in reservoir_idx}

    remaining = [p for p in ordered if p.pair_id not in reservoir_ids]
    strata: dict[str, list[SupportPair]] = {name: [] for name in STRATUM_ORDER}
    for pair in remaining:
        strata[assign_stratum(pair.customer_text)].append(pair)

    # Scale the frozen quotas to the requested size so tests and smaller runs keep the same
    # proportions as the 200-example protocol rather than inventing a second design.
    target = size - reservoir
    scale = target / sum(STRATUM_QUOTAS.values())
    quotas = {name: int(round(quota * scale)) for name, quota in STRATUM_QUOTAS.items()}

    drawn: dict[str, list[SupportPair]] = {}
    shortfall: dict[str, int] = {}
    for name in STRATUM_ORDER:
        available = strata[name]
        want = min(quotas[name], len(available))
        if want < quotas[name]:
            shortfall[name] = quotas[name] - want
        picks = sorted(int(i) for i in rng.choice(len(available), size=want, replace=False)) \
            if want else []
        drawn[name] = [available[i] for i in picks]

    # A stratum that could not fill its quota hands the shortfall to whatever is left, and the
    # manifest records it. Quietly returning fewer than `size` examples would understate the
    # evaluation's size; quietly topping up without saying so would hide the composition.
    selected_ids = reservoir_ids | {p.pair_id for group in drawn.values() for p in group}
    deficit = target - sum(len(g) for g in drawn.values())
    if deficit > 0:
        spare = [p for p in remaining if p.pair_id not in selected_ids]
        if len(spare) < deficit:
            raise SamplingError(
                f"cannot reach {size} examples: {deficit} short after quotas and only "
                f"{len(spare)} unselected pairs remain"
            )
        picks = sorted(int(i) for i in rng.choice(len(spare), size=deficit, replace=False))
        for i in picks:
            drawn.setdefault(FALLBACK_STRATUM, []).append(spare[i])

    records: list[SamplingRecord] = []
    for pair in ordered:
        if pair.pair_id not in reservoir_ids:
            continue
        probability = reservoir / population
        records.append(
            SamplingRecord(
                pair_id=pair.pair_id,
                component="reservoir",
                stratum=assign_stratum(pair.customer_text),
                inclusion_probability=probability,
                weight=1 / probability,
                weak_label=weak_label(pair.customer_text).to_dict(),
            )
        )
    for name in STRATUM_ORDER:
        group = drawn.get(name, [])
        eligible = max(len(strata[name]), len(group))
        for pair in group:
            # Conditional on surviving the reservoir draw, each member of a stratum is equally
            # likely to be picked, so the probability is the stratum's sampling fraction.
            probability = (len(group) / eligible) if eligible else 1.0
            records.append(
                SamplingRecord(
                    pair_id=pair.pair_id,
                    component="stratified",
                    stratum=name,
                    inclusion_probability=probability,
                    weight=1 / probability,
                    weak_label=weak_label(pair.customer_text).to_dict(),
                )
            )

    by_id = {p.pair_id: p for p in ordered}
    chosen_ids = sorted({r.pair_id for r in records})
    candidates = tuple(_to_candidate(by_id[pid]) for pid in chosen_ids)
    records = tuple(sorted(records, key=lambda r: r.pair_id))

    if len(candidates) != size:
        raise SamplingError(
            f"sampler produced {len(candidates)} candidates, expected {size}; refusing to "
            f"emit a set whose size does not match the frozen protocol"
        )

    manifest = {
        "WARNING": (
            "This file records a SAMPLING FRAME, not labels. Every candidate is UNLABELED. "
            "The stratification signal is weakly supervised and is NEVER gold."
        ),
        "protocol": "docs/GOLDEN_SET.md (frozen 2026-09-11, before sampling)",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "git_sha": _git_sha(),
        "brand": BRAND,
        "source_split": "test_pool",
        "seed": seed,
        "population": population,
        "size": size,
        "reservoir_size": reservoir,
        "stratified_size": size - reservoir,
        "taxonomy_version": TAXONOMY.version,
        "taxonomy_hash": TAXONOMY.frozen_hash,
        "label_status": LabelProvenance.UNLABELED.value,
        "human_labelled_count": 0,
        "weak_labels_used_as_gold": False,
        "llm_labels_used_as_gold": False,
        "classifier_predictions_used_as_gold": False,
        "stratification_signal": WEAK_PROVENANCE,
        "stratification_signal_provenance": LabelProvenance.WEAKLY_LABELED.value,
        "strata": {
            "order": list(STRATUM_ORDER),
            "quota": quotas,
            "eligible": {name: len(strata[name]) for name in STRATUM_ORDER},
            "drawn": {name: len(drawn.get(name, [])) for name in STRATUM_ORDER},
            "shortfall": shortfall,
        },
        "reservoir_inclusion_probability": reservoir / population,
    }
    return SamplingResult(candidates=candidates, records=records, manifest=manifest)


def candidates_sha256(candidates: Sequence[GoldenCandidate]) -> str:
    """Content hash used to pin the set once annotation begins."""
    digest = hashlib.sha256()
    for candidate in sorted(candidates, key=lambda c: c.pair_id):
        digest.update(candidate.pair_id.encode("utf-8"))
        digest.update(candidate.customer_message.encode("utf-8"))
    return digest.hexdigest()
