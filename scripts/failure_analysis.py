"""Recompute every count and example ID used in the failure analysis. Reads committed artifacts only.

Inputs (nothing is re-run, no model is called):
    reports/golden_eval/predictions.jsonl
    reports/golden_eval/judge.jsonl
    reports/golden_eval/metrics.json
    data/golden/candidates.jsonl          (for whether prior thread turns exist)

Output:
    reports/golden_eval/failure_analysis_data.json

The narrative in failure_analysis.md quotes these numbers; this script is how they are checked.
It proposes no fix and changes nothing in the agent.

Usage:
    python scripts/failure_analysis.py
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hiver_support.agent.retrieval import _DEFLECTION_TAIL_RE  # noqa: E402

EVAL = ROOT / "reports" / "golden_eval"
ESCALATION_SENSITIVE = {"billing_and_subscription", "repair_order_replacement"}


def _load(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def main() -> None:
    predictions = _load(EVAL / "predictions.jsonl")
    judged = {(j["pair_id"], j["system"]): j for j in _load(EVAL / "judge.jsonl")}
    candidates = {c["pair_id"]: c for c in _load(ROOT / "data" / "golden" / "candidates.jsonl")}
    metrics = json.loads((EVAL / "metrics.json").read_text(encoding="utf-8"))["metrics"]

    by = {s: {r["pair_id"]: r for r in predictions if r["system"] == s}
          for s in ("agent_llm", "agent_template", "baseline_b")}
    agent = by["agent_llm"]
    has_thread = lambda p: bool(candidates[p]["context"])  # noqa: E731
    judge = lambda p: judged.get((p, "agent_llm"), {})  # noqa: E731

    unsafe = sorted(p for p, r in agent.items() if r["gold"]["should_escalate"] and not r["escalate"])
    har_auto = sorted(p for p, r in agent.items()
                      if r["gold"]["expected_resolution_kind"] == "human_action_required" and not r["escalate"])
    context_miss = sorted(p for p, r in agent.items()
                          if not r["gold"]["context_sufficient"] and r["context_sufficient"])
    context_overflag = sorted(p for p, r in agent.items()
                              if r["gold"]["context_sufficient"] and not r["context_sufficient"])
    answered = sorted(p for p, r in agent.items() if not r["escalate"])
    deflecting = sorted(p for p in answered if judge(p).get("deflects"))
    dm_style = sorted(p for p in deflecting if "dm" in (agent[p]["reply"] or "").lower().split()
                      or " dm" in (agent[p]["reply"] or "").lower())
    false_escalations = sorted(p for p, r in agent.items() if not r["gold"]["should_escalate"] and r["escalate"])
    generator_failed = sorted(p for p, r in agent.items() if r["reason"] == "generator_failed")
    intent_errors = sorted(p for p, r in agent.items() if r["intent"] != r["gold"]["intent"])
    unclear_errors = sorted(p for p in intent_errors if agent[p]["intent"] == "other_unclear")
    gold_sensitive = sorted(p for p, r in agent.items() if r["gold"]["intent"] in ESCALATION_SENSITIVE)
    security = metrics["agent_llm"]["all"]["security_sensitive"]
    context = metrics["agent_llm"]["all"]["context_insufficient"]

    def thread_share(ids):
        return {"with_thread_context": sum(has_thread(p) for p in ids), "of": len(ids)}

    data = {
        "_source": "committed artifacts only; no re-run, no model call",
        "base_rate_thread_context": thread_share(list(agent)),
        "F1_human_action_auto_handled": {
            "unsafe_auto_handles": len(unsafe),
            "unsafe_with_gold_human_action_required": sum(
                agent[p]["gold"]["expected_resolution_kind"] == "human_action_required" for p in unsafe),
            "human_action_required_auto_handled_any_escalation_label": len(har_auto),
            "unsafe_also_auto_handled_by_agent_template": sum(not by["agent_template"][p]["escalate"] for p in unsafe),
            "unsafe_also_auto_handled_by_baseline_b": sum(not by["baseline_b"][p]["escalate"] for p in unsafe),
            "unsafe_gold_intent": dict(Counter(agent[p]["gold"]["intent"] for p in unsafe).most_common()),
            "unsafe_gold_intent_escalation_sensitive": sum(agent[p]["gold"]["intent"] in ESCALATION_SENSITIVE for p in unsafe),
            "unsafe_model_confidence_ge_0_8": sum((agent[p]["model_confidence"] or 0) >= 0.8 for p in unsafe),
            "unsafe_judge_mean_relevance": round(sum(judge(p)["relevance"] for p in unsafe) / len(unsafe), 2),
            "unsafe_judge_mean_helpfulness": round(sum(judge(p)["helpfulness"] for p in unsafe) / len(unsafe), 2),
            "unsafe_group": dict(Counter(agent[p]["group"] for p in unsafe)),
            "unsafe_ids": unsafe,
        },
        "F2_context_misses": {
            "gold_context_insufficient": context["tp"] + context["fn"],
            "detected": context["tp"],
            "missed": context["fn"],
            "recall": context["recall"],
            "misses_that_became_unsafe_auto_handles": len(set(context_miss) & set(unsafe)),
            "missed_thread_context": thread_share(context_miss),
            "ids": context_miss,
        },
        "F3_deflection": {
            "answered": len(answered),
            "judged_deflecting": len(deflecting),
            "dm_style": len(dm_style),
            "link_only": len(deflecting) - len(dm_style),
            "dm_style_reply_matches_shared_deflection_regex": sum(
                bool(_DEFLECTION_TAIL_RE.search(agent[p]["reply"])) for p in dm_style),
            "dm_style_evidence_matches_shared_deflection_regex": sum(
                any(_DEFLECTION_TAIL_RE.search(e["resolution_text"]) for e in agent[p]["evidence"]) for p in dm_style),
            "dm_style_evidence_mentions_dm": sum(
                any(" dm" in e["resolution_text"].lower() for e in agent[p]["evidence"]) for p in dm_style),
            "overlap_with_unsafe": len(set(deflecting) & set(unsafe)),
            "judge_mean_groundedness_dm_style": round(sum(judge(p)["groundedness"] for p in dm_style) / len(dm_style), 2),
            "dm_style_ids": dm_style,
            "link_only_ids": sorted(set(deflecting) - set(dm_style)),
        },
        "F4_over_escalation": {
            "false_escalations": len(false_escalations),
            "gold_automatable": sum(not r["gold"]["should_escalate"] for r in agent.values()),
            "false_escalation_rate": metrics["agent_llm"]["all"]["routing"]["false_escalation_rate"],
            "reasons": dict(Counter(agent[p]["reason"] for p in false_escalations).most_common()),
            "context_overflags": len(context_overflag),
            "context_overflag_thread_context": thread_share(context_overflag),
            "security_tp_fp_fn": [security["tp"], security["fp"], security["fn"]],
            "security_precision": security["precision"],
            "false_escalations_from_generator_failure": len(set(false_escalations) & set(generator_failed)),
        },
        "F5_intent_errors": {
            "errors": len(intent_errors),
            "accuracy": metrics["agent_llm"]["all"]["intent"]["accuracy"],
            "predicted_other_unclear_errors": len(unclear_errors),
            "other_unclear_errors_thread_context": thread_share(unclear_errors),
            "top_confusions": [[g, p, n] for (g, p), n in Counter(
                (agent[x]["gold"]["intent"], agent[x]["intent"]) for x in intent_errors).most_common(5)],
            "unsafe_with_wrong_intent": sum(agent[p]["intent"] != agent[p]["gold"]["intent"] for p in unsafe),
            "gold_escalation_sensitive": len(gold_sensitive),
            "gold_escalation_sensitive_predicted_sensitive": sum(agent[p]["intent"] in ESCALATION_SENSITIVE for p in gold_sensitive),
            "gold_escalation_sensitive_auto_handled": sum(not agent[p]["escalate"] for p in gold_sensitive),
        },
        "provider_empty_responses": {
            "count": len(generator_failed),
            "generator_calls": 115,
            "gold_should_escalate_true": sum(agent[p]["gold"]["should_escalate"] for p in generator_failed),
            "gold_should_escalate_false": sum(not agent[p]["gold"]["should_escalate"] for p in generator_failed),
            "template_generator_would_auto_handle": sum(not by["agent_template"][p]["escalate"] for p in generator_failed),
            "ids": generator_failed,
        },
        "evaluation_limitations": {
            "unsafe_by_group": {
                "assisted": [sum(agent[p]["group"] == "assisted" for p in unsafe),
                             sum(r["group"] == "assisted" for r in agent.values())],
                "blind": [sum(agent[p]["group"] == "blind" for p in unsafe),
                          sum(r["group"] == "blind" for r in agent.values())],
            },
            "gold_human_action_required_but_should_escalate_false": sum(
                r["gold"]["expected_resolution_kind"] == "human_action_required" and not r["gold"]["should_escalate"]
                for r in agent.values()),
            "unsafe_thread_context": thread_share(unsafe),
        },
    }
    out = EVAL / "failure_analysis_data.json"
    out.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
