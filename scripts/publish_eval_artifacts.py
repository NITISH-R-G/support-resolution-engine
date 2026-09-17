"""Publish a text-free copy of an evaluation run for committing.

``evaluate_golden.py`` writes full-text outputs to ``data/local/`` because predictions and judge
rationales contain customer messages, brand replies and retrieved evidence. This copies a run to
a publishable directory with every such string replaced by its sha256
(``hiver_support.golden.textfree``), plus the reply/evidence features failure analysis needs.
metrics.json, summary.md and run_context.json contain no tweet text and are copied unchanged.

Usage:
    python scripts/publish_eval_artifacts.py data/local/golden_eval_run reports/golden_eval_new
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hiver_support.agent.retrieval import _DEFLECTION_TAIL_RE  # noqa: E402
from hiver_support.golden.textfree import redact_judgement, redact_prediction  # noqa: E402

JSONL_REDACTORS = {
    "predictions.jsonl": lambda row: redact_prediction(row, _DEFLECTION_TAIL_RE),
    "judge.jsonl": redact_judgement,
    "judge_claude_partial_402.jsonl": redact_judgement,
}
COPIED = ("metrics.json", "summary.md", "run_context.json")


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_bytes(("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n").encode("utf-8"))


def publish(source: Path, target: Path) -> dict:
    target.mkdir(parents=True, exist_ok=True)
    written = {}
    for name, redactor in JSONL_REDACTORS.items():
        if (source / name).exists():
            rows = [redactor(row) for row in read_jsonl(source / name)]
            write_jsonl(target / name, rows)
            written[name] = len(rows)
    for name in COPIED:
        if (source / name).exists() and (source / name).resolve() != (target / name).resolve():
            shutil.copyfile(source / name, target / name)
            written[name] = "copied"
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("target", type=Path)
    args = parser.parse_args()
    print(json.dumps(publish(args.source, args.target), indent=2))


if __name__ == "__main__":
    main()
