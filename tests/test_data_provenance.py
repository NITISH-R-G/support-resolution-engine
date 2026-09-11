"""Enforce the data-provenance boundaries in `docs/DATA_PROVENANCE.md`.

These are repository hygiene tests rather than behaviour tests. They exist because the
separation between synthetic fixtures and real corpus data is a claim we make in the report,
and a claim worth making is worth enforcing mechanically. Two of the seven public
repositories audited in `PUBLIC_REPO_COMPARISON.md` committed hundreds of megabytes of raw
corpus data (one, the same 95 MB file twice), which is the failure this prevents.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

DATA_EXTENSIONS = {".csv", ".tsv", ".parquet", ".jsonl", ".pkl", ".joblib", ".npy", ".faiss", ".zip"}

# Data artifacts that SHOULD be committed. The guard's purpose is to keep *corpus* data out,
# not to ban the extension:
#   data/golden/   - hand-made labels; small, and irreproducible by a reviewer any other way
#   reports/       - derived analysis artifacts; small, carry provenance, and regenerable
#                    from the corpus via scripts/analyse_brands.py
#   tests/fixtures/- committed test fixtures
# Size is policed separately by test_no_tracked_file_exceeds_a_sane_size, so an oversized
# artifact still fails even when its path is allowed here.
ALLOWED_DATA_PATHS = ("data/golden/", "reports/", "tests/fixtures/")


def _tracked_files() -> list[str]:
    result = subprocess.run(
        ["git", "ls-files"], cwd=ROOT, capture_output=True, text=True, check=True
    )
    return [line for line in result.stdout.splitlines() if line]


class TestNoCorpusDataIsCommitted:
    def test_no_bulk_data_files_are_tracked(self):
        offenders = [
            path
            for path in _tracked_files()
            if Path(path).suffix.lower() in DATA_EXTENSIONS
            and not path.startswith(ALLOWED_DATA_PATHS)
        ]
        assert offenders == [], (
            f"data files are tracked by git: {offenders}. Raw and derived corpus data must "
            f"stay out of the repository; reviewers reproduce it via scripts/fetch_data.py."
        )

    def test_raw_data_directory_is_gitignored(self):
        gitignore = (ROOT / ".gitignore").read_text(encoding="utf-8")
        for pattern in ("data/raw/", "data/processed/"):
            assert pattern in gitignore, f"{pattern} must be gitignored"

    def test_no_tracked_file_exceeds_a_sane_size(self):
        """Guards against a corpus or a model pickle arriving by accident."""
        limit_bytes = 2 * 1024 * 1024
        oversized = [
            (path, (ROOT / path).stat().st_size)
            for path in _tracked_files()
            if (ROOT / path).exists() and (ROOT / path).stat().st_size > limit_bytes
        ]
        assert oversized == [], f"tracked files over 2 MB: {oversized}"


class TestFixturesAreSynthetic:
    """The suite must not silently start depending on real data being present."""

    # The single sanctioned exception. Real-data validation must read the corpus; naming it
    # here keeps the fixture/real-data boundary explicit instead of letting it erode.
    REAL_DATA_TEST = "test_real_data.py"

    @pytest.mark.parametrize(
        "reader", ["read_csv", "read_parquet", "read_json", "np.load", "joblib.load"]
    )
    def test_no_test_reads_a_data_file_from_disk(self, reader):
        exempt = {Path(__file__).name, self.REAL_DATA_TEST}
        offenders = [
            path.name
            for path in (ROOT / "tests").glob("test_*.py")
            if path.name not in exempt and reader in path.read_text(encoding="utf-8")
        ]
        assert offenders == [], (
            f"tests read data from disk via {reader}: {offenders}. Every fixture must be "
            f"constructed in memory so the suite proves nothing about data it has not seen."
        )

    def test_the_real_data_test_module_skips_when_the_corpus_is_absent(self):
        """A reviewer without the 493 MB corpus must see skips, never silent passes."""
        source = (ROOT / "tests" / self.REAL_DATA_TEST).read_text(encoding="utf-8")
        assert "pytestmark" in source and "skipif" in source

    def test_pickle_is_never_loaded_anywhere_in_the_project(self):
        """Pickle loading is arbitrary code execution; a reviewer should never face one."""
        sources = list((ROOT / "src").rglob("*.py")) + list((ROOT / "tests").glob("*.py"))
        offenders = [
            path.name
            for path in sources
            if path.name != Path(__file__).name and "pickle.load" in path.read_text(encoding="utf-8")
        ]
        assert offenders == []


class TestVerificationManifestStaysHonest:
    """The manifest must not drift from reality; it is a claim, not decoration."""

    @staticmethod
    def _manifest() -> dict:
        import json

        return json.loads((ROOT / "VERIFICATION.json").read_text(encoding="utf-8"))

    def test_manifest_test_count_is_internally_consistent(self):
        """The per-file breakdown counts COLLECTED tests, which is passing plus skipped.

        Conflating collected with passing is how a manifest starts overstating what was
        demonstrated; a skipped test is not a pass (docs/DATA_PROVENANCE.md).
        """
        tests = self._manifest()["tests"]
        declared = sum(tests["breakdown"].values())
        assert declared == tests["total_collected"], (
            "VERIFICATION.json breakdown does not sum to its own declared collected total"
        )
        assert tests["total_collected"] == tests["total_passing"] + tests["skipped"], (
            "collected must equal passing + skipped"
        )

    def test_manifest_lists_exactly_the_test_files_that_exist(self):
        """Catches the realistic drift: a test file added without updating the manifest."""
        declared = set(self._manifest()["tests"]["breakdown"])
        actual = {path.name for path in (ROOT / "tests").glob("test_*.py")}
        assert declared == actual, (
            f"VERIFICATION.json is stale. Missing: {sorted(actual - declared)}; "
            f"stale entries: {sorted(declared - actual)}"
        )

    def test_manifest_dataset_claims_are_internally_consistent(self):
        """The manifest records what was demonstrated at checkpoint time.

        It is deliberately NOT an assertion about the machine running the suite: a reviewer's
        fresh clone has no corpus, and that must not make the recorded history look false.
        What must hold is internal consistency — you cannot have processed rows without
        having downloaded and verified the file.
        """
        dataset = self._manifest()["dataset"]
        if dataset["twcs_rows_processed"] > 0:
            assert dataset["twcs_downloaded"] is True
            assert dataset["schema_validated_against_real_file"] is True
        if not dataset["twcs_downloaded"]:
            assert dataset["twcs_rows_processed"] == 0

    def test_manifest_never_claims_evaluation_that_has_not_happened(self):
        """Guards the failure mode this project audits others for: unearned claims."""
        evaluation = self._manifest()["evaluation"]
        if not evaluation["golden_set_created"]:
            assert evaluation["golden_set_human_labelled"] is False
            assert evaluation["baselines_scored"] is False
            assert evaluation["judge_human_agreement_measured"] is False
            assert evaluation["any_metric_reported"] is False

    def test_manifest_declares_no_copied_code_or_labels(self):
        provenance = self._manifest()["provenance"]
        assert provenance["code_copied_from_public_hiver_repos"] is False
        assert provenance["data_copied_from_public_hiver_repos"] is False
        assert provenance["labels_copied_from_public_hiver_repos"] is False


class TestACandidateSetIsNeverReportedAsAGoldenSet:
    """The distinction this whole milestone rests on.

    200 unlabelled candidates are not a golden set. The manifest must be incapable of
    implying otherwise, because "golden set: 200" in a report that means "200 rows exist" is
    precisely the overstatement this project audits other submissions for.
    """

    @staticmethod
    def _manifest() -> dict:
        import json

        return json.loads((ROOT / "VERIFICATION.json").read_text(encoding="utf-8"))

    def test_a_built_candidate_set_does_not_make_a_golden_set(self):
        evaluation = self._manifest()["evaluation"]
        if evaluation["golden_candidate_set_built"] and not evaluation["golden_set_created"]:
            assert evaluation["golden_human_labelled_count"] == 0
            assert evaluation["golden_candidates_label_status"] == "UNLABELED"

    def test_claiming_a_golden_set_requires_human_labels(self):
        evaluation = self._manifest()["evaluation"]
        if evaluation["golden_set_created"]:
            assert evaluation["golden_human_labelled_count"] > 0
            assert evaluation["golden_set_human_labelled"] is True

    def test_no_substitute_for_human_labels_is_ever_declared(self):
        evaluation = self._manifest()["evaluation"]
        assert evaluation["weak_labels_used_as_gold"] is False
        assert evaluation["llm_labels_used_as_gold"] is False
        assert evaluation["classifier_predictions_used_as_gold"] is False

    def test_the_candidate_count_matches_the_file_when_it_exists(self):
        candidates = ROOT / "data" / "golden" / "candidates.jsonl"
        if not candidates.exists():
            import pytest

            pytest.skip("golden candidate set not built on this machine")
        lines = [l for l in candidates.read_text(encoding="utf-8").splitlines() if l.strip()]
        assert len(lines) == self._manifest()["evaluation"]["golden_candidates"]

    def test_every_committed_candidate_declares_itself_unlabelled(self):
        import json as _json

        candidates = ROOT / "data" / "golden" / "candidates.jsonl"
        if not candidates.exists():
            import pytest

            pytest.skip("golden candidate set not built on this machine")
        for line in candidates.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = _json.loads(line)
            assert record["label_status"] == "UNLABELED"
            assert "intent" not in record


class TestNoCredentialReachesTheRepository:
    """The LLM layer introduced a second class of secret. Neither may ever be committed."""

    def test_the_env_file_is_gitignored(self):
        ignored = subprocess.run(
            ["git", "check-ignore", ".env"], cwd=ROOT, capture_output=True, text=True
        )
        assert ignored.returncode == 0, ".env is not gitignored"

    def test_the_llm_cache_is_gitignored(self):
        ignored = subprocess.run(
            ["git", "check-ignore", "cache/llm/x.json"], cwd=ROOT, capture_output=True, text=True
        )
        assert ignored.returncode == 0, "cache/llm/ is not gitignored"

    def test_no_tracked_file_contains_an_api_key_shaped_string(self):
        import re

        pattern = re.compile(r"sk-[A-Za-z0-9\-_]{20,}")
        offenders = []
        for path in _tracked_files():
            full = ROOT / path
            if not full.exists() or full.suffix in {".png", ".jpg", ".pkl"}:
                continue
            try:
                text = full.read_text(encoding="utf-8")
            except (UnicodeDecodeError, OSError):
                continue
            if pattern.search(text) and "test_llm_provider" not in path:
                offenders.append(path)
        assert offenders == [], f"possible API keys in tracked files: {offenders}"

    def test_the_example_env_file_ships_with_no_value_set(self):
        example = (ROOT / ".env.example").read_text(encoding="utf-8")
        for line in example.splitlines():
            if line.startswith(("OPENROUTER_API_KEY", "ANTHROPIC_API_KEY", "OPENAI_API_KEY")):
                assert line.strip().endswith("="), f"a key value is committed: {line}"
