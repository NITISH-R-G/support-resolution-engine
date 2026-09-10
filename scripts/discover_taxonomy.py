"""Exploratory discovery for the AppleSupport intent taxonomy (Milestone 3, Step 1).

**This script produces evidence, not labels.** Its clusters are an aid for a human reading
exemplars; no cluster assignment ever becomes a gold label. Three of the seven public
repositories audited in `docs/PUBLIC_REPO_COMPARISON.md` let exploratory model output become
their answer key, and their headline numbers measure nothing as a result.

**Discovery runs on the TRAIN split only.** The golden set is drawn from the held-out test
pool (`SPEC.md` §9.1), so clustering the whole corpus would fit the label design to the very
data the system is later evaluated on. Restricting discovery to train keeps the test pool
genuinely unseen at design time.

Everything is seeded and every parameter is recorded in the output artifact.

Usage:
    python scripts/discover_taxonomy.py                 # full train split
    python scripts/discover_taxonomy.py --limit 400000  # quick pass over fewer raw rows
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
from hiver_support.data.pii import mask_pii  # noqa: E402
from hiver_support.data.reply_classify import classify_reply  # noqa: E402
from hiver_support.data.split import temporal_split  # noqa: E402
from hiver_support.data.threads import (  # noqa: E402
    extract_support_pairs,
    identify_brand,
    parse_tweets,
    reconstruct_conversations,
)
from hiver_support.leakage import normalise_for_comparison  # noqa: E402

BRAND = "AppleSupport"
CORPUS = ROOT / "data" / "raw" / "twcs" / "twcs.csv"
INTERIM = ROOT / "data" / "interim"
REPORTS = ROOT / "reports"

RANDOM_SEED = 20260910
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # local; no API, no cost
EMBED_SAMPLE = 20_000
CLUSTER_RANGE = (8, 10, 12, 14, 16, 20, 24)
EXEMPLARS_PER_CLUSTER = 12
NEAR_DUPLICATE_THRESHOLD = 0.90

# --- pattern probes -----------------------------------------------------------------------
# Explicit and auditable. These COUNT phenomena in the data; they do not assign labels.

MULTI_INTENT_RE = re.compile(
    r"\b(and also|also,|as well as|plus,? )\b|\?.*\?", re.IGNORECASE
)
ESCALATION_RE = re.compile(
    r"\b(refund|charged|charge|billing|invoice|overcharg|double.?charg|money back|"
    r"fraud|hacked|unauthoriz|unauthoris|stolen|scam|breach|privacy|"
    r"lawsuit|legal|lawyer|sue|attorney|"
    r"supervisor|manager|complaint|escalate|useless|disgrace|worst)\b",
    re.IGNORECASE,
)
QUESTION_RE = re.compile(r"\?")


def _git_sha() -> str:
    try:
        return (
            subprocess.run(
                ["git", "rev-parse", "--short", "HEAD"],
                cwd=ROOT,
                capture_output=True,
                text=True,
            ).stdout.strip()
            or "unknown"
        )
    except Exception:  # pragma: no cover - provenance is best-effort
        return "unknown"


def _digest(path: Path, limit_bytes: int = 64 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        h.update(handle.read(limit_bytes))
    return h.hexdigest()


def load_brand_pairs(limit: int | None) -> list:
    """Reconstruct the corpus and return this brand's customer/support pairs.

    Cached to `data/interim/` (git-ignored) because reconstruction takes ~90s and discovery
    is inherently iterative.
    """
    INTERIM.mkdir(parents=True, exist_ok=True)
    cache = INTERIM / f"{BRAND.lower()}_pairs_limit_{limit or 'all'}.pkl"
    if cache.exists():
        import pickle

        print(f"Loading cached pairs from {cache}")
        with cache.open("rb") as handle:
            return pickle.load(handle)

    print(f"Loading {CORPUS} ...")
    started = time.time()
    frame = pd.read_csv(CORPUS, dtype=str, keep_default_na=False, na_values=[""], nrows=limit)
    print(f"  {len(frame):,} records in {time.time() - started:.1f}s")

    tweets = parse_tweets(frame)
    del frame
    gc.collect()
    conversations = reconstruct_conversations(tweets)
    del tweets
    gc.collect()

    pairs = []
    for conversation in conversations:
        if identify_brand(conversation) == BRAND:
            pairs.extend(extract_support_pairs(conversation, BRAND))
    print(f"  {len(pairs):,} {BRAND} pairs")

    import pickle

    with cache.open("wb") as handle:
        pickle.dump(pairs, handle)
    return pairs


def _sample_indices(n: int, size: int, seed: int) -> np.ndarray:
    if n <= size:
        return np.arange(n)
    return np.sort(np.random.default_rng(seed).choice(n, size=size, replace=False))


def basic_statistics(pairs: list) -> dict:
    """Volume, duplication, length and question-shape statistics over customer messages."""
    texts = [normalise_text(p.customer_text, brand=BRAND) for p in pairs]
    comparison = [normalise_for_comparison(p.customer_text) for p in pairs]
    non_empty = [t for t in comparison if t]

    word_counts = np.array([len(t.split()) for t in texts])
    replies = [classify_reply(normalise_text(p.support_text, brand=BRAND)) for p in pairs]

    return {
        "pairs": len(pairs),
        "distinct_customers": len({p.customer_tweet.author_id for p in pairs}),
        "distinct_conversations": len({p.conversation_id for p in pairs}),
        "exact_duplicate_rate": round(1 - len(set(non_empty)) / max(len(non_empty), 1), 4),
        "empty_after_normalisation": len(comparison) - len(non_empty),
        "words_mean": round(float(word_counts.mean()), 2),
        "words_median": int(np.median(word_counts)),
        "words_p10": int(np.percentile(word_counts, 10)),
        "words_p90": int(np.percentile(word_counts, 90)),
        "very_short_rate_lt5_words": round(float((word_counts < 5).mean()), 4),
        "contains_question_rate": round(
            float(np.mean([bool(QUESTION_RE.search(t)) for t in texts])), 4
        ),
        "multi_intent_signal_rate": round(
            float(np.mean([bool(MULTI_INTENT_RE.search(t)) for t in texts])), 4
        ),
        "escalation_signal_rate": round(
            float(np.mean([bool(ESCALATION_RE.search(t)) for t in texts])), 4
        ),
        "reply_deflection_rate": round(np.mean([r.is_deflection for r in replies]), 4),
        "reply_substantive_rate": round(np.mean([r.is_substantive for r in replies]), 4),
        "reply_actionable_rate": round(np.mean([r.is_actionable for r in replies]), 4),
        "with_context_rate": round(np.mean([bool(p.context) for p in pairs]), 4),
    }


def lexical_profile(texts: list[str], top_n: int = 40) -> dict:
    """Top unigrams and bigrams by document frequency — a cheap first look at what is asked."""
    from sklearn.feature_extraction.text import CountVectorizer

    vectorizer = CountVectorizer(
        stop_words="english", ngram_range=(1, 2), min_df=5, max_features=20_000, binary=True
    )
    matrix = vectorizer.fit_transform(texts)
    frequencies = np.asarray(matrix.sum(axis=0)).ravel()
    vocabulary = np.array(vectorizer.get_feature_names_out())
    order = np.argsort(-frequencies)

    unigrams = [
        (vocabulary[i], int(frequencies[i])) for i in order if " " not in vocabulary[i]
    ][:top_n]
    bigrams = [(vocabulary[i], int(frequencies[i])) for i in order if " " in vocabulary[i]][
        :top_n
    ]
    return {
        "documents": len(texts),
        "top_unigrams": unigrams,
        "top_bigrams": bigrams,
    }


def embed(texts: list[str]) -> np.ndarray:
    """Local sentence embeddings. No API, no cost, deterministic given the pinned model."""
    from sentence_transformers import SentenceTransformer

    print(f"  embedding {len(texts):,} messages with {EMBED_MODEL} (local) ...")
    started = time.time()
    model = SentenceTransformer(EMBED_MODEL)
    vectors = model.encode(
        texts, batch_size=256, show_progress_bar=False, normalize_embeddings=True
    )
    print(f"    done in {time.time() - started:.1f}s")
    return np.asarray(vectors, dtype=np.float32)


def cluster_sweep(vectors: np.ndarray, texts: list[str]) -> dict:
    """Cluster at several k and report separation, so k is chosen on evidence not by fiat."""
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    probe = _sample_indices(len(vectors), 5_000, RANDOM_SEED + 3)
    results = []
    models = {}

    for k in CLUSTER_RANGE:
        kmeans = KMeans(n_clusters=k, random_state=RANDOM_SEED, n_init=10)
        labels = kmeans.fit_predict(vectors)
        shares = np.array(sorted(Counter(labels).values(), reverse=True)) / len(labels)
        silhouette = float(silhouette_score(vectors[probe], labels[probe], metric="cosine"))
        results.append(
            {
                "k": k,
                "silhouette_cosine": round(silhouette, 4),
                "largest_share": round(float(shares[0]), 4),
                "smallest_share": round(float(shares[-1]), 4),
                "clusters_at_or_above_3pct": int((shares >= 0.03).sum()),
                "clusters_below_1pct": int((shares < 0.01).sum()),
            }
        )
        models[k] = (kmeans, labels)
        print(f"    k={k:>2}  silhouette={silhouette:.4f}  >=3%: {(shares >= 0.03).sum()}")

    return {"sweep": results, "models": models}


def cluster_exemplars(
    vectors: np.ndarray, labels: np.ndarray, centres: np.ndarray, texts: list[str], pairs: list
) -> list[dict]:
    """Nearest-to-centroid messages per cluster, for a human to read and name.

    Exemplars are PII-masked before they reach any document.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer

    tfidf = TfidfVectorizer(stop_words="english", min_df=3, ngram_range=(1, 2))
    matrix = tfidf.fit_transform(texts)
    vocabulary = np.array(tfidf.get_feature_names_out())

    out = []
    for cluster in range(centres.shape[0]):
        member_idx = np.flatnonzero(labels == cluster)
        if member_idx.size == 0:
            continue

        similarity = vectors[member_idx] @ centres[cluster]
        nearest = member_idx[np.argsort(-similarity)[:EXEMPLARS_PER_CLUSTER]]

        centroid_tfidf = np.asarray(matrix[member_idx].mean(axis=0)).ravel()
        top_terms = vocabulary[np.argsort(-centroid_tfidf)[:12]].tolist()

        replies = [
            classify_reply(normalise_text(pairs[i].support_text, brand=BRAND))
            for i in member_idx
        ]
        escalation_hits = sum(1 for i in member_idx if ESCALATION_RE.search(texts[i]))

        out.append(
            {
                "cluster": int(cluster),
                "size": int(member_idx.size),
                "share": round(float(member_idx.size / len(labels)), 4),
                "top_terms": top_terms,
                "reply_deflection_rate": round(np.mean([r.is_deflection for r in replies]), 3),
                "reply_actionable_rate": round(np.mean([r.is_actionable for r in replies]), 3),
                "escalation_signal_rate": round(escalation_hits / member_idx.size, 3),
                "mean_words": round(
                    float(np.mean([len(texts[i].split()) for i in member_idx])), 1
                ),
                "exemplars": [
                    {
                        "pair_id": pairs[i].pair_id,
                        "customer": mask_pii(texts[i]).text[:260],
                        "brand_reply": mask_pii(
                            normalise_text(pairs[i].support_text, brand=BRAND)
                        ).text[:260],
                    }
                    for i in nearest
                ],
            }
        )
    return sorted(out, key=lambda c: -c["size"])


def hard_case_probes(texts: list[str], pairs: list) -> dict:
    """Count and sample the cases a taxonomy has to have an explicit policy for."""
    rng = np.random.default_rng(RANDOM_SEED + 5)

    def sample(indices: list[int], n: int = 8) -> list[dict]:
        if not indices:
            return []
        chosen = rng.choice(indices, size=min(n, len(indices)), replace=False)
        return [
            {
                "pair_id": pairs[i].pair_id,
                "customer": mask_pii(texts[i]).text[:220],
                "brand_reply": mask_pii(
                    normalise_text(pairs[i].support_text, brand=BRAND)
                ).text[:220],
            }
            for i in sorted(chosen)
        ]

    very_short = [i for i, t in enumerate(texts) if len(t.split()) < 5]
    multi = [i for i, t in enumerate(texts) if MULTI_INTENT_RE.search(t)]
    escalation = [i for i, t in enumerate(texts) if ESCALATION_RE.search(t)]
    no_question = [
        i
        for i, t in enumerate(texts)
        if not QUESTION_RE.search(t) and len(t.split()) >= 5
    ]

    return {
        "very_short_lt5_words": {"count": len(very_short), "examples": sample(very_short)},
        "multi_intent_signal": {"count": len(multi), "examples": sample(multi)},
        "escalation_signal": {"count": len(escalation), "examples": sample(escalation)},
        "statement_not_question": {
            "count": len(no_question),
            "examples": sample(no_question),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    if not CORPUS.exists():
        sys.exit(f"Corpus not found at {CORPUS}. Run: python scripts/fetch_data.py")

    started = time.time()
    pairs = load_brand_pairs(args.limit)

    splits = temporal_split(pairs)
    train = list(splits.train)
    print(
        f"\nSplit: train={len(train):,}  dev={len(splits.dev):,}  "
        f"test_pool={len(splits.test_pool):,}  dropped={len(splits.dropped):,}"
    )
    print("Discovery uses the TRAIN split only; the test pool stays unseen at design time.\n")

    texts = [normalise_text(p.customer_text, brand=BRAND) for p in train]

    print("Basic statistics ...")
    statistics = basic_statistics(train)

    print("Lexical profile ...")
    lexical = lexical_profile(texts)

    print("Semantic clustering ...")
    embed_idx = _sample_indices(len(train), EMBED_SAMPLE, RANDOM_SEED + 1)
    embed_texts = [texts[i] for i in embed_idx]
    embed_pairs = [train[i] for i in embed_idx]
    vectors = embed(embed_texts)

    sweep = cluster_sweep(vectors, embed_texts)
    models = sweep.pop("models")

    exemplars = {
        str(k): cluster_exemplars(
            vectors, labels, kmeans.cluster_centers_, embed_texts, embed_pairs
        )
        for k, (kmeans, labels) in models.items()
        if k in (10, 12, 14, 16)
    }

    print("Hard-case probes ...")
    probes = hard_case_probes(texts, train)

    artifact = {
        "provenance": {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "script": "scripts/discover_taxonomy.py",
            "git_sha": _git_sha(),
            "brand": BRAND,
            "corpus_path": str(CORPUS.relative_to(ROOT)).replace("\\", "/"),
            "corpus_sha256_first_64mb": _digest(CORPUS),
            "corpus_modified": False,
            "random_seed": RANDOM_SEED,
            "row_limit": args.limit,
            "embedding_model": EMBED_MODEL,
            "embedding_sample": int(len(embed_idx)),
            "cluster_range": list(CLUSTER_RANGE),
            "split_manifest": splits.manifest,
            "split_used_for_discovery": "train",
            "elapsed_seconds": round(time.time() - started, 1),
            "note": (
                "Exploratory evidence only. No cluster assignment is a label. Discovery is "
                "restricted to the train split so the held-out test pool, from which the "
                "golden set is drawn, remains unseen at taxonomy-design time."
            ),
        },
        "statistics": statistics,
        "lexical": lexical,
        "cluster_sweep": sweep["sweep"],
        "cluster_exemplars": exemplars,
        "hard_case_probes": probes,
    }

    REPORTS.mkdir(exist_ok=True)
    out = REPORTS / "taxonomy_discovery.json"
    out.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
    print(f"\nWrote {out}")
    print(f"Elapsed {artifact['provenance']['elapsed_seconds']}s")


if __name__ == "__main__":
    main()
