"""The repository stores no tweet text, and nothing may put it back.

Release remediation removed every customer message, brand reply and customer-derived model output
from tracked files. The text is rebuilt locally by ``scripts/materialize_text.py``. Two guards
keep it that way:

* **Structural** (always runs): committed data and report JSON may not carry a value under a
  known text field, unless that value is a hash or a redaction marker. The ``probes`` sections of
  the smoke reports hold synthetic probe sentences and are exempt.
* **Content** (real data; skips without the corpus): the shared detector
  (``scripts/audit/tweet_text.py``) finds no corpus tweet text in any tracked file.

Both have negative controls, so a vacuous pass is itself a failure.
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
TEXT_FIELDS = {"message", "reply", "customer_message", "text", "customer", "brand_reply",
               "rationale", "customer_text", "resolution_text", "reason_detail", "draft", "response"}
REDACTED = re.compile(r"^\[(tweet-)?text redacted: .*\]$")
HASH = re.compile(r"^[0-9a-f]{64}$")
SCOPE = ("data/", "reports/")


def _tracked_json_files() -> list[str]:
    files = subprocess.run(["git", "ls-files"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.split()
    return [f for f in files if f.startswith(SCOPE) and f.endswith((".json", ".jsonl"))]


def _violations(value, path: str = "$", inside_probes: bool = False) -> list[str]:
    found = []
    if isinstance(value, dict):
        for key, item in value.items():
            if (key in TEXT_FIELDS and isinstance(item, str) and item.strip() and not inside_probes
                    and not REDACTED.match(item) and not HASH.match(item)):
                found.append(f"{path}.{key}")
            found.extend(_violations(item, f"{path}.{key}", inside_probes or key == "probes"))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            found.extend(_violations(item, f"{path}[{index}]", inside_probes))
    return found


def _file_violations(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8")
    if path.suffix == ".jsonl":
        found = []
        for number, line in enumerate(text.splitlines(), start=1):
            if line.strip():
                found.extend(_violations(json.loads(line), f"line{number}"))
        return found
    return _violations(json.loads(text))


class TestStructural:
    def test_no_committed_data_or_report_file_carries_text_fields(self):
        offenders = {}
        for rel in _tracked_json_files():
            found = _file_violations(ROOT / rel)
            if found:
                offenders[rel] = found[:5]
        assert offenders == {}, f"text fields in committed files: {offenders}"

    def test_the_committed_golden_candidates_are_text_free_but_complete(self):
        records = [json.loads(l) for l in (ROOT / "data/golden/candidates.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
        assert len(records) == 200
        for record in records:
            assert "customer_message" not in record
            assert set(record["text_sha256"]) == {"v1", "v2"}
            assert all("text" not in turn for turn in record["context"])

    def test_negative_control_a_planted_text_field_is_caught(self, tmp_path):
        planted = tmp_path / "planted.jsonl"
        planted.write_text(json.dumps({"pair_id": "1__2", "message": "my phone died again"}) + "\n", encoding="utf-8")
        assert _file_violations(planted) == ["line1.message"]

    def test_negative_control_hashes_and_markers_are_allowed(self, tmp_path):
        clean = tmp_path / "clean.json"
        clean.write_text(json.dumps({
            "message_sha256": "ab" * 32,
            "reply": "[text redacted: sha256=0123456789abcdef]",
            "text": "[tweet-text redacted: tweet_id=1 sha256=0123456789abcdef]",
            "probes": [{"message": "a synthetic probe"}],
        }), encoding="utf-8")
        assert _file_violations(clean) == []


PAIRS = ROOT / "data" / "interim" / "applesupport_pairs_limit_all.pkl"


@pytest.mark.skipif(not PAIRS.exists(), reason="corpus not reconstructed on this machine")
class TestContent:
    @pytest.fixture(scope="class")
    def detector(self):
        sys.path.insert(0, str(ROOT / "scripts" / "audit"))
        import tweet_text

        return tweet_text

    @pytest.fixture(scope="class")
    def index(self, detector):
        return detector.TweetIndex()

    def test_no_tracked_file_contains_corpus_tweet_text(self, detector, index):
        files = subprocess.run(["git", "ls-files"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.split()
        offenders = {}
        for rel in files:
            path = ROOT / rel
            if path.is_file():
                matches, _ = detector.scan_bytes(rel, path.read_bytes(), index)
                if matches:
                    offenders[rel] = len(matches)
        assert offenders == {}, f"corpus tweet text found: {offenders}"

    def test_negative_control_a_planted_tweet_is_detected(self, detector, index):
        # A real golden message, from the locally rebuilt v1 text (never committed).
        from hiver_support.golden import paths

        local = paths.local_candidates("v1")
        if not local.exists():
            pytest.skip("golden text not materialised (python scripts/materialize_text.py --golden)")
        messages = [json.loads(l)["customer_message"] for l in local.read_text(encoding="utf-8").splitlines() if l.strip()]
        message = next(m for m in messages if len(detector.words_of(m)) >= 14 and index.match(m))
        matches, _ = detector.scan_bytes("x.json", json.dumps({"note": message}).encode(), index)
        assert matches
