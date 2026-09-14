"""Audit C: negative controls for the safety-critical tests (mutation testing).

Each mutation deliberately breaks one safeguard. The tests covering it must FAIL; a mutation the
suite does not catch is a test gap. Every file is restored in a finally block, and the working
tree is checked against git afterwards.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

MUTATIONS = [
    ("M1 security gate disabled", "src/hiver_support/agent/agent.py",
     "        if prediction.security_sensitive:", "        if False and prediction.security_sensitive:",
     ["tests/test_agent.py", "tests/test_llm_generation.py"]),
    ("M2 context gate disabled", "src/hiver_support/agent/agent.py",
     "        if not prediction.context_sufficient:", "        if False and not prediction.context_sufficient:",
     ["tests/test_agent.py", "tests/test_llm_generation.py"]),
    ("M3 policy-intent gate disabled", "src/hiver_support/agent/agent.py",
     "        if TAXONOMY.get(prediction.intent).escalation_sensitive:", "        if False:",
     ["tests/test_agent.py", "tests/test_llm_generation.py"]),
    ("M4 model-requested escalation ignored", "src/hiver_support/agent/agent.py",
     "        if draft.model_should_escalate:", "        if False and draft.model_should_escalate:",
     ["tests/test_llm_generation.py"]),
    ("M5 grounding result ignored by agent", "src/hiver_support/agent/agent.py",
     "        if not report.passed:", "        if False and not report.passed:",
     ["tests/test_agent.py", "tests/test_llm_generation.py"]),
    ("M6 grounding validator always passes", "src/hiver_support/agent/grounding.py",
     "    return GroundingReport(passed=not violations, violations=tuple(violations))",
     "    return GroundingReport(passed=True, violations=())",
     ["tests/test_grounding.py", "tests/test_agent.py"]),
    ("M7 policy violations ignored by agent", "src/hiver_support/agent/agent.py",
     "        if policy.fatal:", "        if False and policy.fatal:",
     ["tests/test_agent.py", "tests/test_llm_generation.py", "tests/test_routing_invariants.py"]),
    ("M8 generated deflection not detected", "src/hiver_support/agent/policy.py",
     "    if _DEFLECTION_TAIL_RE.search(reply):", "    if False and _DEFLECTION_TAIL_RE.search(reply):",
     ["tests/test_policy_validator.py"]),
    ("M9 unknown evidence label accepted", "src/hiver_support/agent/generation.py",
     "    if unknown:", "    if False and unknown:",
     ["tests/test_llm_generation.py"]),
    ("M10 load_gold accepts partial set", "src/hiver_support/golden/store.py",
     '    if require_complete and not stats["complete"]:', "    if False:",
     ["tests/test_golden_store.py", "tests/test_golden_retraction.py"]),
    ("M11 annotation accepts non-human provenance", "src/hiver_support/golden/schema.py",
     "        if self.provenance is not LabelProvenance.HUMAN_LABELED:", "        if False:",
     ["tests/test_golden_schema.py", "tests/test_golden_store.py"]),
    ("M12 id-overlap leakage guard disabled", "src/hiver_support/leakage.py",
     "        if overlap:", "        if False and overlap:",
     ["tests/test_leakage.py"]),
    ("M14 retriever failure no longer escalates", "src/hiver_support/agent/agent.py",
     "except Exception as exc:  # noqa: BLE001 - any retrieval failure must fail closed",
     "except ZeroDivisionError as exc:  # noqa: BLE001 - any retrieval failure must fail closed",
     ["tests/test_dependency_failures.py"]),
    ("M15 classifier failure no longer escalates", "src/hiver_support/agent/agent.py",
     "except Exception as exc:  # noqa: BLE001 - any classifier failure must fail closed",
     "except ZeroDivisionError as exc:  # noqa: BLE001 - any classifier failure must fail closed",
     ["tests/test_dependency_failures.py"]),
    ("M13 false auto-handle rate diluted by all traffic", "src/hiver_support/evaluation/metrics.py",
     '        "false_auto_handle_rate": _ratio(unsafe, len(should)),',
     '        "false_auto_handle_rate": _ratio(unsafe, len(pred_escalate)),',
     ["tests/test_evaluation_metrics.py"]),
]


def run_tests(tests: list[str]) -> tuple[bool, str]:
    existing = [t for t in tests if (ROOT / t).exists()]
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "-x", "-q", "--color=no", "-p", "no:cacheprovider", *existing],
        cwd=ROOT, capture_output=True, text=True,
    )
    failed = [l for l in proc.stdout.splitlines() if l.startswith("FAILED") or l.startswith("ERROR")]
    return proc.returncode == 0, (failed[0][:110] if failed else ""), existing


def main():
    rows = []
    for name, rel, old, new, tests in MUTATIONS:
        path = ROOT / rel
        original_bytes = path.read_bytes()
        original = original_bytes.decode("utf-8")
        count = original.count(old)
        if count != 1:
            rows.append((name, f"SKIPPED - anchor found {count} times", ""))
            continue
        try:
            path.write_bytes(original_bytes.replace(old.encode(), new.encode()))
            passed, first_failure, existing = run_tests(tests)
        finally:
            path.write_bytes(original_bytes)
        verdict = "CAUGHT" if not passed else "NOT CAUGHT  <-- test gap"
        rows.append((name, verdict, first_failure or f"(ran {', '.join(Path(t).name for t in existing)})"))
    width = max(len(r[0]) for r in rows)
    for name, verdict, detail in rows:
        print(f"{name:<{width}}  {verdict}")
        print(f"{'':<{width}}    {detail}")
    diff = subprocess.run(["git", "status", "--porcelain", "src"], cwd=ROOT, capture_output=True, text=True).stdout
    print("\nsource restored (git status src clean):", diff.strip() == "")


if __name__ == "__main__":
    main()
