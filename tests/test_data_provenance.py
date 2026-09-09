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

# Golden labels are the one data artifact that SHOULD be committed once it exists: it is
# small, hand-made by us, and the reviewer cannot reproduce it any other way.
ALLOWED_DATA_PATHS = ("data/golden/", "tests/fixtures/")


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

    @pytest.mark.parametrize(
        "reader", ["read_csv", "read_parquet", "read_json", "np.load", "joblib.load"]
    )
    def test_no_test_reads_a_data_file_from_disk(self, reader):
        offenders = [
            path.name
            for path in (ROOT / "tests").glob("test_*.py")
            # This file names the readers it forbids, so it must not scan itself.
            if path.name != Path(__file__).name and reader in path.read_text(encoding="utf-8")
        ]
        assert offenders == [], (
            f"tests read data from disk via {reader}: {offenders}. Every fixture must be "
            f"constructed in memory so the suite proves nothing about data it has not seen."
        )

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
        manifest = self._manifest()
        declared = sum(manifest["tests"]["breakdown"].values())
        assert declared == manifest["tests"]["total_passing"], (
            "VERIFICATION.json breakdown does not sum to its own declared total"
        )

    def test_manifest_lists_exactly_the_test_files_that_exist(self):
        """Catches the realistic drift: a test file added without updating the manifest."""
        declared = set(self._manifest()["tests"]["breakdown"])
        actual = {path.name for path in (ROOT / "tests").glob("test_*.py")}
        assert declared == actual, (
            f"VERIFICATION.json is stale. Missing: {sorted(actual - declared)}; "
            f"stale entries: {sorted(declared - actual)}"
        )

    def test_manifest_claims_no_real_data_while_none_is_present(self):
        manifest = self._manifest()
        corpus_present = (ROOT / "data" / "raw").exists()
        if not corpus_present:
            assert manifest["dataset"]["twcs_downloaded"] is False
            assert manifest["dataset"]["twcs_rows_processed"] == 0
            assert manifest["tests"]["real_data_used_in_tests"] is False

    def test_manifest_declares_no_copied_code_or_labels(self):
        provenance = self._manifest()["provenance"]
        assert provenance["code_copied_from_public_hiver_repos"] is False
        assert provenance["data_copied_from_public_hiver_repos"] is False
        assert provenance["labels_copied_from_public_hiver_repos"] is False
