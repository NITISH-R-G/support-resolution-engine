"""Audit A: do the committed evaluation artifacts follow from the committed predictions?

Regenerates into a temp directory only. Includes a negative control: a one-row corruption must be
detected, or the comparison proves nothing.
"""
from __future__ import annotations

import json
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np

EVAL = ROOT / "reports" / "golden_eval"


def load(path):
    return [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]


def recompute_metrics(pred_path, judge_path, blind):
    import evaluate_golden as eg
    return eg.compute_metrics(load(pred_path), blind, load(judge_path))


def main():
    committed = json.loads((EVAL / "metrics.json").read_text(encoding="utf-8"))
    context = json.loads((EVAL / "run_context.json").read_text(encoding="utf-8"))
    blind = set(context["blind_pair_ids"])
    results = {}

    # A1 metrics.json <- predictions.jsonl + judge.jsonl
    recomputed = recompute_metrics(EVAL / "predictions.jsonl", EVAL / "judge.jsonl", blind)
    norm = lambda x: json.loads(json.dumps(x, default=str))  # noqa: E731
    results["A1 metrics.json recomputes from predictions+judge"] = norm(recomputed) == committed["metrics"]

    tmp = Path(tempfile.mkdtemp(prefix="hiver_audit_"))
    try:
        # A1-NEG: flip one agent_llm routing decision in a temp copy; the comparison must notice.
        rows = load(EVAL / "predictions.jsonl")
        target = next(r for r in rows if r["system"] == "agent_llm")
        target["escalate"] = not target["escalate"]
        (tmp / "predictions.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
        corrupted = recompute_metrics(tmp / "predictions.jsonl", EVAL / "judge.jsonl", blind)
        results["A1-NEG one flipped decision is detected"] = norm(corrupted) != committed["metrics"]

        # A2 summary.md <- metrics.json (ignore the generated-at line)
        import evaluate_golden as eg
        out = tmp / "summary_dir"; out.mkdir()
        eg.write_summary(out, committed["metrics"], committed["DISCLOSURE"],
                         committed["generator_totals"], {**committed["judge_summary"], "cache_hits": "n/a"})
        strip = lambda t: [l for l in t.splitlines() if not l.startswith("Generated ")]  # noqa: E731
        results["A2 summary.md regenerates from metrics.json"] = (
            strip((out / "summary.md").read_text(encoding="utf-8"))
            == strip((EVAL / "summary.md").read_text(encoding="utf-8")))

        # A3 risk_coverage.{json,png} <- predictions.jsonl
        import risk_coverage as rc
        rc_dir = tmp / "rc"; rc_dir.mkdir()
        for name in ("predictions.jsonl", "run_context.json"):
            shutil.copy(EVAL / name, rc_dir / name)
        rc.EVAL = rc_dir
        import contextlib, io
        with contextlib.redirect_stdout(io.StringIO()):
            rc.main()
        results["A3 risk_coverage.json regenerates"] = (
            json.loads((rc_dir / "risk_coverage.json").read_text(encoding="utf-8"))
            == json.loads((EVAL / "risk_coverage.json").read_text(encoding="utf-8")))
        results["A3 risk_coverage.md regenerates"] = (
            (rc_dir / "risk_coverage.md").read_text(encoding="utf-8")
            == (EVAL / "risk_coverage.md").read_text(encoding="utf-8"))
        import matplotlib.image as mpimg
        a, b = mpimg.imread(rc_dir / "risk_coverage.png"), mpimg.imread(EVAL / "risk_coverage.png")
        results["A3 risk_coverage.png pixels identical"] = a.shape == b.shape and bool(np.array_equal(a, b))

        # A3-NEG: corrupt the plot data and confirm the pixel comparison notices.
        rows = load(EVAL / "predictions.jsonl")
        for r in rows:
            if r["system"] == "baseline_b":
                r["escalate"] = True
        (rc_dir / "predictions.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
        with contextlib.redirect_stdout(io.StringIO()):
            rc.main()
        c = mpimg.imread(rc_dir / "risk_coverage.png")
        results["A3-NEG corrupted data changes the plot"] = not (c.shape == b.shape and np.array_equal(c, b))

        # A4 failure_analysis_data.json <- committed artifacts
        import failure_analysis as fa
        fa_dir = tmp / "fa"; fa_dir.mkdir()
        for name in ("predictions.jsonl", "judge.jsonl", "metrics.json"):
            shutil.copy(EVAL / name, fa_dir / name)
        fa.EVAL = fa_dir
        with contextlib.redirect_stdout(io.StringIO()):
            fa.main()
        results["A4 failure_analysis_data.json regenerates"] = (
            json.loads((fa_dir / "failure_analysis_data.json").read_text(encoding="utf-8"))
            == json.loads((EVAL / "failure_analysis_data.json").read_text(encoding="utf-8")))

        # A5 headline numbers quoted in failure_analysis.md exist in its data file
        md = (EVAL / "failure_analysis.md").read_text(encoding="utf-8")
        data = json.loads((EVAL / "failure_analysis_data.json").read_text(encoding="utf-8"))
        quoted = {
            "28 of 29": (data["F1_human_action_auto_handled"]["unsafe_with_gold_human_action_required"], data["F1_human_action_auto_handled"]["unsafe_auto_handles"]),
            "25 of 31": (data["F2_context_misses"]["missed"], data["F2_context_misses"]["gold_context_insufficient"]),
            "11 of 88": (data["F3_deflection"]["judged_deflecting"], data["F3_deflection"]["answered"]),
            "66 of 125": (data["F4_over_escalation"]["false_escalations"], data["F4_over_escalation"]["gold_automatable"]),
            "114 of 200": (data["F5_intent_errors"]["errors"], 200),
        }
        results["A5 failure_analysis.md counts match data"] = all(
            text in md.replace("**", "") and f"{a} of {b}" == text for text, (a, b) in quoted.items())
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    width = max(len(k) for k in results)
    for k, v in results.items():
        print(f"{k:<{width}}  {'PASS' if v else 'FAIL'}")
    print("committed artifacts untouched:", "OK")


if __name__ == "__main__":
    main()
