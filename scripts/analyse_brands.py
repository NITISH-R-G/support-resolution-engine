"""Compute the frozen brand-selection profile for every candidate brand.

Implements `docs/SPEC.md` §3.2. Every feature here is a **descriptive statistic over the
corpus**: no agent is run, no model is trained, and no downstream performance influences the
result. That separation is the point — selecting the brand that yields the best metric would
be a garden-of-forking-paths error guaranteeing an inflated, non-replicating number
(`DECISION_LOG.md` D14).

The raw corpus is opened read-only and never modified.

Usage:
    python scripts/analyse_brands.py
    python scripts/analyse_brands.py --limit 300000    # quick pass for development
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import re
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hiver_support.data.normalise import normalise_text  # noqa: E402
from hiver_support.data.reply_classify import classify_reply  # noqa: E402
from hiver_support.data.threads import (  # noqa: E402
    extract_support_pairs,
    identify_brand,
    parse_tweets,
    reconstruct_conversations,
)
from hiver_support.leakage import normalise_for_comparison  # noqa: E402

CORPUS = ROOT / "data" / "raw" / "twcs" / "twcs.csv"
REPORTS = ROOT / "reports"

RANDOM_SEED = 20260910

# Only brands with real volume are profiled; below this a temporal split is degenerate.
MIN_PAIRS_TO_PROFILE = 1_000

# Expensive features (clustering, nearest-neighbour) run on a seeded sample per brand so
# every brand is measured at the same cost and the numbers stay comparable.
FEATURE_SAMPLE = 3_000
N_CLUSTERS = 12
RARE_INTENT_BAND = (0.01, 0.05)
MIN_INTENT_SHARE = 0.03
NEAR_DUPLICATE_THRESHOLD = 0.90

# Escalation-sensitivity lexicon. Kept explicit and auditable: it drives brand selection, so
# a reviewer must be able to see and challenge it. Reply classification lives in the tested
# module hiver_support.data.reply_classify.

ESCALATION_RE = re.compile(
    r"\b(refund|charged|charge|billing|bill|invoice|overcharg|double.?charg|money back|"
    r"fraud|hacked|unauthoriz|unauthoris|stolen|scam|breach|"
    r"lawsuit|legal|solicitor|lawyer|sue|attorney|ombudsman|"
    r"injur|unsafe|danger|assault|threat|harass|discriminat|racist|"
    r"supervisor|manager|complaint|escalate|cancel my (account|subscription|membership))\b",
    re.IGNORECASE,
)



def _git_sha() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True
        ).stdout.strip() or "unknown"
    except Exception:  # pragma: no cover - provenance is best-effort
        return "unknown"


def _file_digest(path: Path, limit_bytes: int = 64 * 1024 * 1024) -> str:
    """Hash a prefix of the corpus. Full-file hashing of 493 MB adds minutes for no gain."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        digest.update(handle.read(limit_bytes))
    return digest.hexdigest()


def load_pairs(limit: int | None) -> tuple[dict[str, list], dict]:
    """Reconstruct the corpus and group customer/support pairs by brand."""
    print(f"Loading {CORPUS} ...")
    started = time.time()
    frame = pd.read_csv(
        CORPUS, dtype=str, keep_default_na=False, na_values=[""], nrows=limit
    )
    rows = len(frame)
    print(f"  {rows:,} records in {time.time() - started:.1f}s")

    tweets = parse_tweets(frame)
    print(f"  parsed {len(tweets):,} tweets ({rows - len(tweets):,} unusable rows dropped)")
    del frame
    gc.collect()

    conversations = reconstruct_conversations(tweets)
    print(f"  reconstructed {len(conversations):,} conversations")
    del tweets
    gc.collect()

    by_brand: dict[str, list] = {}
    no_brand = 0
    for conversation in conversations:
        brand = identify_brand(conversation)
        if brand is None:
            no_brand += 1
            continue
        pairs = extract_support_pairs(conversation, brand)
        if pairs:
            by_brand.setdefault(brand, []).extend(pairs)

    stats = {
        "records_read": rows,
        "conversations": len(conversations),
        "conversations_without_brand": no_brand,
        "brands_seen": len(by_brand),
        "total_pairs": sum(len(v) for v in by_brand.values()),
    }
    print(f"  {stats['total_pairs']:,} pairs across {stats['brands_seen']} brands")
    return by_brand, stats


def _sample(values: list, size: int, seed: int = RANDOM_SEED) -> list:
    if len(values) <= size:
        return values
    rng = np.random.default_rng(seed)
    index = rng.choice(len(values), size=size, replace=False)
    return [values[i] for i in sorted(index)]


def _reply_features(pairs: list) -> dict:
    """Deflection, substance and actionability rates over brand replies."""
    deflection = substantive = actionable = non_english = 0
    for pair in pairs:
        reply = normalise_text(pair.support_text, brand=pair.brand)
        result = classify_reply(reply)
        deflection += result.is_deflection
        substantive += result.is_substantive
        actionable += result.is_actionable
        non_english += not result.is_english
    total = len(pairs)
    return {
        "dm_deflection_rate": round(deflection / total, 4),
        "substantive_resolution_rate": round(substantive / total, 4),
        "actionable_resolution_rate": round(actionable / total, 4),
        "non_english_reply_rate": round(non_english / total, 4),
    }


def _duplicate_features(pairs: list) -> dict:
    texts = [normalise_for_comparison(p.customer_text) for p in pairs]
    non_empty = [t for t in texts if t]
    exact_duplicates = len(non_empty) - len(set(non_empty))
    return {
        "exact_duplicate_rate": round(exact_duplicates / max(len(non_empty), 1), 4),
        "empty_after_normalisation": len(texts) - len(non_empty),
    }


def _intent_and_retrieval_features(pairs: list) -> dict:
    """Topic spread and within-brand retrievability, both on a seeded sample."""
    from sklearn.cluster import KMeans
    from sklearn.feature_extraction.text import TfidfVectorizer

    sampled = _sample(pairs, FEATURE_SAMPLE)
    texts = [normalise_for_comparison(p.customer_text) for p in sampled]
    texts = [t for t in texts if len(t.split()) >= 3]
    if len(texts) < N_CLUSTERS * 5:
        return {"insufficient_text_for_clustering": True}

    word_vectorizer = TfidfVectorizer(
        max_features=5_000, stop_words="english", min_df=3, ngram_range=(1, 2)
    )
    matrix = word_vectorizer.fit_transform(texts)

    kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=RANDOM_SEED, n_init=5)
    labels = kmeans.fit_predict(matrix)
    shares = np.array([count for _, count in sorted(Counter(labels).items())]) / len(labels)

    entropy = float(-(shares * np.log(shares + 1e-12)).sum())
    low, high = RARE_INTENT_BAND

    # Retrievability: how similar is a message to its nearest *other* message in the same
    # brand? A brand whose questions never resemble each other cannot be served by
    # retrieval, however good the model is.
    char_vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2)
    char_matrix = char_vectorizer.fit_transform(texts)
    probe = _sample(list(range(len(texts))), 500, seed=RANDOM_SEED + 1)
    similarities = (char_matrix[probe] @ char_matrix.T).toarray()
    for row, index in enumerate(probe):
        similarities[row, index] = -1.0  # exclude self-match
    nearest = similarities.max(axis=1)

    return {
        "clusters_sampled": len(texts),
        "distinct_intents_at_3pct": int((shares >= MIN_INTENT_SHARE).sum()),
        "largest_intent_share": round(float(shares.max()), 4),
        "intent_entropy": round(entropy, 4),
        "intent_entropy_normalised": round(entropy / np.log(N_CLUSTERS), 4),
        "rare_intent_clusters": int(((shares >= low) & (shares < high)).sum()),
        "rare_intent_pairs_estimated": int(
            shares[(shares >= low) & (shares < high)].sum() * len(pairs)
        ),
        "near_duplicate_rate": round(float((nearest >= NEAR_DUPLICATE_THRESHOLD).mean()), 4),
        "retrieval_nn_similarity_median": round(float(np.median(nearest)), 4),
        "retrieval_nn_similarity_p25": round(float(np.percentile(nearest, 25)), 4),
    }


def _thread_and_time_features(pairs: list) -> dict:
    lengths = [len(p.context) + 2 for p in pairs]
    times = sorted(p.customer_tweet.created_at for p in pairs)
    span_days = (times[-1] - times[0]).days if len(times) > 1 else 0
    months = len({(t.year, t.month) for t in times})
    return {
        "with_context_rate": round(sum(1 for p in pairs if p.context) / len(pairs), 4),
        "mean_thread_length": round(float(np.mean(lengths)), 2),
        "median_thread_length": int(np.median(lengths)),
        "first_seen": times[0].isoformat(),
        "last_seen": times[-1].isoformat(),
        "span_days": span_days,
        "distinct_months": months,
    }


def _escalation_features(pairs: list) -> dict:
    hits = sum(1 for p in pairs if ESCALATION_RE.search(p.customer_text))
    return {
        "escalation_sensitive_rate": round(hits / len(pairs), 4),
        "escalation_sensitive_count": hits,
    }


def profile_brand(brand: str, pairs: list) -> dict:
    profile = {
        "brand": brand,
        "pair_count": len(pairs),
        "distinct_customers": len({p.customer_tweet.author_id for p in pairs}),
        "distinct_conversations": len({p.conversation_id for p in pairs}),
    }
    profile |= _reply_features(pairs)
    profile |= _duplicate_features(pairs)
    profile |= _thread_and_time_features(pairs)
    profile |= _escalation_features(pairs)
    profile |= _intent_and_retrieval_features(pairs)

    # Feature 13: the conjunction that actually binds reply quality. A brand can look fine on
    # each component while very few pairs satisfy all of them together.
    profile["usable_grounding_evidence_rate"] = round(
        profile["actionable_resolution_rate"] * (1 - profile["exact_duplicate_rate"]), 4
    )
    profile["usable_grounding_evidence_pairs"] = int(
        profile["usable_grounding_evidence_rate"] * len(pairs)
    )
    return profile


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=None, help="read only the first N rows")
    args = parser.parse_args()

    if not CORPUS.exists():
        sys.exit(f"Corpus not found at {CORPUS}. Run: python scripts/fetch_data.py")

    started = time.time()
    by_brand, corpus_stats = load_pairs(args.limit)

    candidates = {b: p for b, p in by_brand.items() if len(p) >= MIN_PAIRS_TO_PROFILE}
    print(f"\nProfiling {len(candidates)} brands with >= {MIN_PAIRS_TO_PROFILE:,} pairs ...")

    profiles = []
    for index, (brand, pairs) in enumerate(
        sorted(candidates.items(), key=lambda kv: -len(kv[1])), start=1
    ):
        print(f"  [{index}/{len(candidates)}] {brand} ({len(pairs):,} pairs)")
        profiles.append(profile_brand(brand, pairs))

    provenance = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "script": "scripts/analyse_brands.py",
        "git_sha": _git_sha(),
        "corpus_path": str(CORPUS.relative_to(ROOT)),
        "corpus_size_bytes": CORPUS.stat().st_size,
        "corpus_sha256_first_64mb": _file_digest(CORPUS),
        "corpus_modified": False,
        "random_seed": RANDOM_SEED,
        "row_limit": args.limit,
        "feature_sample_size": FEATURE_SAMPLE,
        "n_clusters": N_CLUSTERS,
        "min_pairs_to_profile": MIN_PAIRS_TO_PROFILE,
        "elapsed_seconds": round(time.time() - started, 1),
        "corpus_stats": corpus_stats,
        "note": (
            "Descriptive statistics only. No model was trained and no agent was run; no "
            "downstream performance influenced these numbers. See DECISION_LOG.md D14."
        ),
    }

    REPORTS.mkdir(exist_ok=True)
    output = {"provenance": provenance, "profiles": profiles}
    (REPORTS / "brand_profiles.json").write_text(
        json.dumps(output, indent=2) + "\n", encoding="utf-8"
    )
    print(f"\nWrote {REPORTS / 'brand_profiles.json'} ({len(profiles)} profiles)")
    print(f"Elapsed {provenance['elapsed_seconds']}s")


if __name__ == "__main__":
    main()
