"""Apply the pre-registered brand-selection criteria and emit the comparison artifact.

Reads `reports/brand_profiles.json` (descriptive corpus statistics only) and applies the
filters and rubric frozen in `docs/SPEC.md` §3.2.3 **before** any of these numbers existed.

No model output, agent behaviour, or downstream metric is an input here. Selecting the brand
that yields the best score would be a garden-of-forking-paths error guaranteeing an inflated,
non-replicating result (`DECISION_LOG.md` D14).

Usage:
    python scripts/select_brand.py
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

# Note: pandas 2.2 emits a spurious ChainedAssignmentError from inside its own DataFrame
# .assign() implementation when copy-on-write is enabled, so CoW is left at its default.
# All frames here are built with .assign(), which is the copy-on-write-safe idiom regardless.

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"

# --- pre-registered thresholds (SPEC.md §3.2.3, frozen before profiling) -----------------
MIN_PAIRS = 5_000
MIN_INTENTS_AT_3PCT = 6
MIN_USABLE_GROUNDING_PAIRS = 1_500
MIN_ESCALATION_SENSITIVE = 300
MIN_DISTINCT_MONTHS = 6
# "Always escalate" must be a real baseline: neither unbeatable nor irrelevant.
ESCALATION_RATE_BAND = (0.01, 0.40)

# Equal weights across the five rubric dimensions. Deliberate: any other weighting would be
# chosen by us, and choosing weights after seeing the profiles is how a "principled" rubric
# becomes a way to justify a preferred answer.
RUBRIC_WEIGHTS = {
    "volume": 0.2,
    "intent_diversity": 0.2,
    "substantive_resolutions": 0.2,
    "retrieval_potential": 0.2,
    "evaluation_coverage": 0.2,
}

TABLE_COLUMNS = [
    ("brand", "Brand"),
    ("pair_count", "Pairs"),
    ("dm_deflection_rate", "Deflect"),
    ("substantive_resolution_rate", "Subst"),
    ("actionable_resolution_rate", "Action"),
    ("usable_grounding_evidence_pairs", "Usable"),
    ("distinct_intents_at_3pct", "Intents"),
    ("intent_entropy_normalised", "Entropy"),
    ("rare_intent_pairs_estimated", "Rare"),
    ("retrieval_nn_similarity_median", "RetrNN"),
    ("escalation_sensitive_count", "EscN"),
    ("escalation_sensitive_rate", "EscRate"),
    ("exact_duplicate_rate", "Dup"),
    ("non_english_reply_rate", "NonEng"),
    ("with_context_rate", "Ctx"),
    ("distinct_months", "Months"),
    ("passes_all", "Pass"),
    ("failed_criteria", "Failed criteria"),
]


def _minmax(series: pd.Series) -> pd.Series:
    span = series.max() - series.min()
    if span == 0:
        return pd.Series(0.5, index=series.index)
    return (series - series.min()) / span


def apply_filters(df: pd.DataFrame) -> pd.DataFrame:
    """Record pass/fail per criterion so rejections are auditable, not merely counted."""
    df = df.assign(
        f_volume=df.pair_count >= MIN_PAIRS,
        f_intents=df.distinct_intents_at_3pct >= MIN_INTENTS_AT_3PCT,
        f_grounding=df.usable_grounding_evidence_pairs >= MIN_USABLE_GROUNDING_PAIRS,
        f_escalation_volume=df.escalation_sensitive_count >= MIN_ESCALATION_SENSITIVE,
        f_escalation_band=df.escalation_sensitive_rate.between(*ESCALATION_RATE_BAND),
        f_temporal=df.distinct_months >= MIN_DISTINCT_MONTHS,
    )
    filters = [c for c in df.columns if c.startswith("f_")]
    return df.assign(
        passes_all=df[filters].all(axis=1),
        failed_criteria=df.apply(
            lambda row: ", ".join(c[2:] for c in filters if not row[c]) or "-", axis=1
        ),
    )


def score(df: pd.DataFrame) -> pd.DataFrame:
    """Score survivors on the five rubric dimensions, equally weighted."""
    # Volume on a log scale: the gap between 5k and 50k pairs matters, between 100k and 150k
    # it does not, and a linear scale would let one very large brand dominate the rubric.
    #
    # Absolute usable pairs, not just the rate: 9,000 usable pairs at a mediocre rate beats a
    # high rate over 400 pairs, because both the golden set and the corpus need volume.
    df = df.assign(
        s_volume=_minmax(np.log10(df.pair_count)),
        s_intent_diversity=_minmax(
            _minmax(df.distinct_intents_at_3pct) + _minmax(df.intent_entropy_normalised)
        ),
        s_substantive_resolutions=_minmax(
            _minmax(np.log10(df.usable_grounding_evidence_pairs.clip(lower=1)))
            + _minmax(df.actionable_resolution_rate)
        ),
        s_retrieval_potential=_minmax(df.retrieval_nn_similarity_median),
        s_evaluation_coverage=_minmax(
            _minmax(np.log10(df.escalation_sensitive_count.clip(lower=1)))
            + _minmax(np.log10(df.rare_intent_pairs_estimated.clip(lower=1)))
        ),
    )
    weighted = sum(df[f"s_{name}"] * weight for name, weight in RUBRIC_WEIGHTS.items())
    return df.assign(rubric_score=weighted.round(4)).sort_values(
        "rubric_score", ascending=False
    )


def _format_cell(key: str, value: object) -> str:
    if key == "brand":
        return f"`{value}`"
    if key == "passes_all":
        return "**YES**" if value else "no"
    if isinstance(value, (bool, np.bool_)):
        return str(bool(value))
    if isinstance(value, float):
        return f"{value:.3f}"
    if isinstance(value, (int, np.integer)):
        return f"{int(value):,}"
    return str(value)


def _write_markdown(df: pd.DataFrame, ranked: pd.DataFrame, selected, decision: dict) -> None:
    """Emit the human-readable comparison, including every rejected brand."""
    prov = decision["provenance"]
    stats = prov["corpus_stats"]
    thresholds = decision["thresholds"]

    parts: list[str] = [
        "# Brand Selection",
        "",
        f"**Selected brand: `{selected.brand}`**",
        "",
        "Criteria and weights were frozen in `docs/SPEC.md` §3.2.3 **before** these numbers",
        "existed. Inputs are descriptive corpus statistics only: no model was trained, no",
        "agent was run, and no downstream performance influenced this choice. There is no",
        "re-selection — if this brand proves hard, that is a reported finding, not a reason",
        "to switch (`DECISION_LOG.md` D14).",
        "",
        "## Provenance",
        "",
        "| Field | Value |",
        "|---|---|",
        f"| Corpus | `{prov['corpus_path']}` |",
        f"| Corpus size | {prov['corpus_size_bytes'] / 1024**2:.0f} MB |",
        f"| Corpus SHA-256 (first 64 MB) | `{prov['corpus_sha256_first_64mb'][:32]}...` |",
        f"| Corpus modified by this project | {prov['corpus_modified']} |",
        f"| Records read | {stats['records_read']:,} |",
        f"| Conversations reconstructed | {stats['conversations']:,} |",
        f"| Customer/support pairs | {stats['total_pairs']:,} |",
        f"| Brands present | {stats['brands_seen']} |",
        f"| Brands profiled (>= 1,000 pairs) | {len(df)} |",
        f"| Random seed | {prov['random_seed']} |",
        f"| Feature sample per brand | {prov['feature_sample_size']:,} |",
        f"| Analysis git SHA | `{prov['git_sha']}` |",
        f"| Generated (UTC) | {prov['generated_at']} |",
        f"| Elapsed | {prov['elapsed_seconds']}s |",
        "",
        "## Pre-registered filters",
        "",
        f"- `pair_count` >= {thresholds['min_pairs']:,}",
        f"- `distinct_intents_at_3pct` >= {thresholds['min_intents_at_3pct']}",
        f"- `usable_grounding_evidence_pairs` >= {thresholds['min_usable_grounding_pairs']:,}",
        f"- `escalation_sensitive_count` >= {thresholds['min_escalation_sensitive']}",
        f"- `escalation_sensitive_rate` within {tuple(thresholds['escalation_rate_band'])}",
        f"- `distinct_months` >= {thresholds['min_distinct_months']}",
        "",
        f"**{decision['brands_passing_filters']} of {decision['brands_profiled']} brands "
        f"passed all six filters.**",
        "",
        "## Survivors, ranked",
        "",
        "| Rank | Brand | Score | Volume | Intent div. | Substantive | Retrieval | Eval cov. |",
        "|---|---|---|---|---|---|---|---|",
    ]

    for rank, row in enumerate(ranked.itertuples(), start=1):
        parts.append(
            f"| {rank} | `{row.brand}` | **{row.rubric_score}** | {row.s_volume:.2f} | "
            f"{row.s_intent_diversity:.2f} | {row.s_substantive_resolutions:.2f} | "
            f"{row.s_retrieval_potential:.2f} | {row.s_evaluation_coverage:.2f} |"
        )

    parts += [
        "",
        "All five dimensions are equally weighted (0.2 each). Component scores are min-max",
        "normalised across survivors, so they are relative to this field, not absolute.",
        "",
        "## Full profile — every brand, including rejected",
        "",
        "Published so a reviewer can see what was traded away, not only the winner's numbers.",
        "",
        "| " + " | ".join(header for _, header in TABLE_COLUMNS) + " |",
        "|" + "---|" * len(TABLE_COLUMNS),
    ]

    for row in df.sort_values("pair_count", ascending=False).itertuples():
        parts.append(
            "| "
            + " | ".join(_format_cell(key, getattr(row, key)) for key, _ in TABLE_COLUMNS)
            + " |"
        )

    parts += [
        "",
        "Column key: **Deflect** = share of replies whose only function is redirecting to",
        "another channel; **Subst** = substantive resolution rate; **Action** = actionable",
        "resolution rate; **Usable** = pairs carrying usable grounding evidence (the",
        "conjunction of actionable and non-duplicate — SPEC §3.2.2 feature 13); **RetrNN** =",
        "median nearest-neighbour similarity within the brand; **EscN/EscRate** =",
        "escalation-sensitive volume and rate; **NonEng** = non-English reply rate.",
        "",
        "Machine-readable equivalents: `brand_profiles.json`, `brand_profiles_all.csv`,",
        "`brand_decision.json`.",
        "",
    ]

    (REPORTS / "brand_selection.md").write_text("\n".join(parts), encoding="utf-8")


def main() -> None:
    profiles_path = REPORTS / "brand_profiles.json"
    if not profiles_path.exists():
        sys.exit(f"{profiles_path} not found. Run: python scripts/analyse_brands.py")

    payload = json.loads(profiles_path.read_text(encoding="utf-8"))
    df = apply_filters(pd.DataFrame(payload["profiles"]))

    survivors = df[df.passes_all].copy()
    if survivors.empty:
        sys.exit(
            "No brand passed the pre-registered filters. The criteria must be revisited "
            "explicitly and the change logged — not silently relaxed."
        )

    ranked = score(survivors)
    selected = ranked.iloc[0]

    decision = {
        "selected_brand": selected.brand,
        "decided_at": datetime.now(timezone.utc).isoformat(),
        "criteria_frozen_in": "docs/SPEC.md 3.2.3",
        "inputs": "reports/brand_profiles.json (descriptive corpus statistics only)",
        "model_performance_used": False,
        "brands_profiled": len(df),
        "brands_passing_filters": len(survivors),
        "thresholds": {
            "min_pairs": MIN_PAIRS,
            "min_intents_at_3pct": MIN_INTENTS_AT_3PCT,
            "min_usable_grounding_pairs": MIN_USABLE_GROUNDING_PAIRS,
            "min_escalation_sensitive": MIN_ESCALATION_SENSITIVE,
            "min_distinct_months": MIN_DISTINCT_MONTHS,
            "escalation_rate_band": list(ESCALATION_RATE_BAND),
        },
        "rubric_weights": RUBRIC_WEIGHTS,
        "ranking": [
            {
                "brand": row.brand,
                "rubric_score": row.rubric_score,
                "components": {
                    name: round(getattr(row, f"s_{name}"), 4) for name in RUBRIC_WEIGHTS
                },
            }
            for row in ranked.itertuples()
        ],
        "provenance": payload["provenance"],
    }

    (REPORTS / "brand_decision.json").write_text(
        json.dumps(decision, indent=2) + "\n", encoding="utf-8"
    )
    df.sort_values("pair_count", ascending=False).to_csv(
        REPORTS / "brand_profiles_all.csv", index=False
    )
    _write_markdown(df, ranked, selected, decision)

    print(f"SELECTED: {selected.brand}")
    print(f"  {len(survivors)} of {len(df)} brands passed the pre-registered filters")
    print(f"  rubric score {selected.rubric_score}")
    print("\nSurvivors:")
    columns = [
        "brand",
        "pair_count",
        "usable_grounding_evidence_pairs",
        "distinct_intents_at_3pct",
        "retrieval_nn_similarity_median",
        "escalation_sensitive_count",
        "rubric_score",
    ]
    print(ranked[columns].to_string(index=False))


if __name__ == "__main__":
    main()
