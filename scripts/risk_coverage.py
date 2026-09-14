"""Risk-coverage curves for every system on the golden set. Measures the tradeoff; picks nothing.

For each system and each confidence score it records, the curve raises a confidence bar on top
of the system's own routing: at threshold t a message is auto-handled only if the system
auto-handled it AND the score is at least t. The first row (t = 0) is the system's committed
operating point; every later row adds abstention.

**No threshold is recommended, and no score is declared best.** Both would be chosen by looking
at the gold set, which turns measurement into tuning. The right operating point depends on how
much worse an unsafe automated reply is than a human handling the message (C_bad / C_human),
which cannot be known from tweets. The curve exposes the tradeoff so that point can be chosen
from that cost, on data that is not the evaluation set.

Reads reports/golden_eval/predictions.jsonl; makes no model calls.

Usage:
    python scripts/risk_coverage.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hiver_support.evaluation.metrics import risk_coverage_curve  # noqa: E402

EVAL = ROOT / "reports" / "golden_eval"
THRESHOLDS = [round(i / 20, 2) for i in range(21)]
TABLE_THRESHOLDS = [round(i / 10, 1) for i in range(11)]
RATIOS = (2, 4, 8, 12, 20)

# Every score each system actually records. None is omitted, because choosing which to show on
# the gold set would be the selection this analysis refuses to make.
CURVES = {
    "agent_llm": ("model_confidence", "intent_confidence", "retrieval_confidence"),
    "agent_template": ("intent_confidence", "retrieval_confidence"),
    "baseline_b": ("intent_confidence", "retrieval_confidence"),
}
SCORE_NOTES = {
    "model_confidence": "self-reported by the generator LLM; uncalibrated",
    "intent_confidence": "TF-IDF classifier probability, trained on weak labels; not calibrated on gold",
    "retrieval_confidence": "top fused retrieval score, min-max normalised per query; relative, not a probability",
}


def _pct(value) -> str:
    return "n/a" if value is None else f"{value * 100:.1f}%"


def _ci(ci) -> str:
    return "" if not ci else f" [{ci[0] * 100:.0f}-{ci[1] * 100:.0f}]"


def main() -> None:
    records = [
        json.loads(line)
        for line in (EVAL / "predictions.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    missing = [r for r in records if "intent_confidence" not in r]
    if missing:
        raise SystemExit(
            f"{len(missing)} predictions lack confidence scores. Re-run:\n"
            "  python scripts/evaluate_golden.py --stage all --offline ..."
        )
    context = json.loads((EVAL / "run_context.json").read_text(encoding="utf-8"))
    blind = set(context["blind_pair_ids"])
    subsets = {
        "all": lambda r: True,
        "blind_human_entered": lambda r: r["pair_id"] in blind,
    }

    results: dict = {}
    for subset, keep in subsets.items():
        results[subset] = {}
        for system in ("agent_llm", "agent_template", "baseline_b", "baseline_a"):
            rows = [r for r in records if r["system"] == system and keep(r)]
            gold = [r["gold"]["should_escalate"] for r in rows]
            auto = [not r["escalate"] for r in rows]
            correct = [r["intent"] == r["gold"]["intent"] for r in rows]
            if system == "baseline_a":
                # Always escalates: one point, not a curve.
                results[subset][system] = {
                    "operating_point_only": risk_coverage_curve(
                        gold, auto, [None] * len(rows), correct, [0.0], RATIOS
                    )
                }
                continue
            results[subset][system] = {
                score: risk_coverage_curve(
                    gold, auto, [r[score] for r in rows], correct, THRESHOLDS, RATIOS
                )
                for score in CURVES[system]
            }

    payload = {
        "WARNING": (
            "No threshold in this file is recommended or optimal. Rows are measurements across "
            "a fixed grid on the gold set. Choosing an operating point here would be tuning on "
            "the evaluation set."
        ),
        "DISCLOSURE": (
            "Gold should_escalate labels on the 160 assisted examples match the Llama "
            "pre-annotation on 159 (accepted in a median 0.4s). All-200 curves measure agreement "
            "with human-accepted pre-annotation; the blind_human_entered subset (40) was labelled "
            "from scratch."
        ),
        "rule": "auto-handled at t iff the system auto-handled it AND score >= t; t=0 ignores the score",
        "definitions": {
            "auto_handle_rate": "auto-handled / all messages (automation)",
            "escalation_rate": "1 - auto_handle_rate",
            "coverage": "auto-handled / messages the human judged automatable",
            "false_auto_handle_rate": "auto-handled / messages the human judged should escalate",
            "unsafe_auto_handle_rate": "should-escalate-but-auto-handled / all messages",
            "selective_risk": "should-escalate-but-auto-handled / auto-handled",
            "intent_accuracy_on_auto_handled": "intent accuracy on the messages actually answered",
            "cost_at_ratio": "mean cost per message: unsafe auto-handle = ratio, escalation = 1, safe auto-handle = 0",
        },
        "score_notes": SCORE_NOTES,
        "thresholds": THRESHOLDS,
        "bootstrap_resamples": 1000,
        "curves": results,
    }
    (EVAL / "risk_coverage.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )

    _write_markdown(results)
    _plot(results)
    print(f"Wrote {EVAL / 'risk_coverage.json'}")
    print(f"Wrote {EVAL / 'risk_coverage.md'}")
    print(f"Wrote {EVAL / 'risk_coverage.png'}")


def _table(rows: list[dict]) -> list[str]:
    lines = [
        "| Threshold | Auto-handle rate | Coverage | False auto-handle rate | Unsafe (all msgs) "
        "| Selective risk | Intent acc (answered) | Escalation rate | Cost@2 | Cost@8 | Cost@20 |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    by_t = {r["threshold"]: r for r in rows}
    for t in TABLE_THRESHOLDS:
        r = by_t[t]
        lines.append(
            f"| {t:.1f} | {_pct(r['auto_handle_rate'])} ({r['auto_handled']}) "
            f"| {_pct(r['coverage'])}{_ci(r['coverage_ci95'])} "
            f"| {_pct(r['false_auto_handle_rate'])}{_ci(r['false_auto_handle_rate_ci95'])} "
            f"| {_pct(r['unsafe_auto_handle_rate'])} ({r['unsafe_auto_handles']}) "
            f"| {_pct(r['selective_risk'])} | {_pct(r['intent_accuracy_on_auto_handled'])} "
            f"| {_pct(r['escalation_rate'])} | {r['cost_at_ratio']['2']:.2f} "
            f"| {r['cost_at_ratio']['8']:.2f} | {r['cost_at_ratio']['20']:.2f} |"
        )
    return lines


def _write_markdown(results: dict) -> None:
    lines = [
        "# Risk-coverage: the automation/safety tradeoff",
        "",
        "**No threshold here is recommended.** Every row is a measurement on a fixed grid. "
        "Picking the best-looking row would be tuning on the evaluation set.",
        "",
        "At threshold t a message is auto-handled only if the system auto-handled it **and** its "
        "confidence score is at least t. Row 0.0 is the system's committed operating point; "
        "higher thresholds add abstention. A threshold can only withhold automation, never add "
        "it, so no curve extends past its system's own automation level.",
        "",
        "The operating point should be chosen from C_bad / C_human, the cost of an unsafe automated "
        "reply relative to a human handling the message. That cost cannot be inferred from tweets, "
        "so the cost columns sweep it (2, 8, 20) instead of assuming one.",
        "",
        "**Disclosure.** On the 160 assisted examples, the gold `should_escalate` label matches the "
        "Llama pre-annotation on 159. All-200 curves measure agreement with human-accepted "
        "pre-annotation. The blind 40 were labelled from scratch.",
        "",
        "Brackets are bootstrap 95% intervals (1,000 resamples). The full 0.05 grid is in "
        "`risk_coverage.json`; tables show every 0.1.",
        "",
        "Score notes:",
    ] + [f"- `{k}`: {v}" for k, v in SCORE_NOTES.items()] + [
        "",
        "![risk-coverage](risk_coverage.png)",
        "",
    ]
    for subset, title in (("all", "All 200"), ("blind_human_entered", "Blind 40 (labelled from scratch)")):
        lines += [f"## {title}", ""]
        a = results[subset]["baseline_a"]["operating_point_only"][0]
        lines += [
            f"**baseline_a (always escalate):** auto-handle rate {_pct(a['auto_handle_rate'])}, "
            f"false auto-handle rate {_pct(a['false_auto_handle_rate'])}, cost 1.00 at every ratio.",
            "",
        ]
        for system, curves in results[subset].items():
            if system == "baseline_a":
                continue
            for score, rows in curves.items():
                lines += [f"### {system} - `{score}`", ""] + _table(rows) + [""]
    (EVAL / "risk_coverage.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _plot(results: dict) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colours = {"agent_llm": "#1f6feb", "agent_template": "#8250df", "baseline_b": "#cf222e"}
    styles = {"model_confidence": "-", "intent_confidence": "--", "retrieval_confidence": ":"}
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    curves = results["all"]

    for system, by_score in curves.items():
        if system == "baseline_a":
            continue
        for score, rows in by_score.items():
            x = [r["auto_handle_rate"] for r in rows]
            y = [r["false_auto_handle_rate"] for r in rows]
            cov = [r["coverage"] for r in rows]
            risk = [r["selective_risk"] if r["selective_risk"] is not None else float("nan") for r in rows]
            label = f"{system} · {score.replace('_confidence', '')}"
            axes[0].plot(x, y, styles[score], color=colours[system], label=label, linewidth=1.6)
            axes[1].plot(cov, risk, styles[score], color=colours[system], label=label, linewidth=1.6)
        op = next(iter(by_score.values()))[0]
        axes[0].scatter([op["auto_handle_rate"]], [op["false_auto_handle_rate"]],
                        color=colours[system], zorder=5, s=45, edgecolor="black")
        if op["selective_risk"] is not None:
            axes[1].scatter([op["coverage"]], [op["selective_risk"]],
                            color=colours[system], zorder=5, s=45, edgecolor="black")

    axes[0].scatter([0], [0], marker="s", color="black", s=50, zorder=6, label="baseline_a · always escalate")
    axes[0].set_xlabel("Auto-handle rate  (more automation →)")
    axes[0].set_ylabel("False auto-handle rate  (should-escalate messages auto-handled)")
    axes[0].set_title("Automation vs unsafe automation")
    axes[1].set_xlabel("Coverage  (automatable messages auto-handled →)")
    axes[1].set_ylabel("Selective risk  (auto-handled that should have escalated)")
    axes[1].set_title("Coverage vs risk among answered messages")
    for ax in axes:
        ax.set_xlim(-0.02, 1.0)
        ax.set_ylim(-0.02, 1.0)
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=7.5, loc="upper left")
    fig.suptitle(
        "Risk-coverage on the golden set (all 200). Dots = committed operating points. "
        "No threshold is selected.",
        fontsize=10,
    )
    fig.text(
        0.5, 0.005,
        "Gold should_escalate on 159/160 assisted examples equals the Llama pre-annotation; "
        "curves measure agreement with human-accepted pre-annotation.",
        ha="center", fontsize=8, style="italic",
    )
    fig.tight_layout(rect=(0, 0.03, 1, 0.95))
    fig.savefig(EVAL / "risk_coverage.png", dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    main()
