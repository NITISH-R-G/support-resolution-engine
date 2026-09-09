"""Kaggle credential detection.

Kaggle supports several authentication mechanisms and has added new ones over time. The
first version of this project's detector recognised only the two legacy methods and would
have refused to run on a machine whose Kaggle CLI authenticates perfectly well — a genuine
integration bug, found by a real setup rather than by these tests.

Detection returns the *name* of the mechanism found, never the credential itself. Every test
below uses a fake home directory and a scrubbed environment, so no real token is read and
none can leak into output.
"""

from __future__ import annotations

import json

import pytest

from hiver_support.kaggle_auth import credential_source

ALL_KAGGLE_VARS = ("KAGGLE_API_TOKEN", "KAGGLE_USERNAME", "KAGGLE_KEY", "KAGGLE_CONFIG_DIR")

FAKE_TOKEN = "fake-token-value-not-a-real-credential"


@pytest.fixture
def clean_env(monkeypatch):
    """Remove every Kaggle variable so a developer's real credentials cannot affect a test."""
    for name in ALL_KAGGLE_VARS:
        monkeypatch.delenv(name, raising=False)
    return monkeypatch


@pytest.fixture
def fake_home(tmp_path):
    """An empty home directory, so the real ~/.kaggle is never consulted."""
    return tmp_path


class TestCurrentTokenAuth:
    def test_detects_kaggle_api_token(self, clean_env, fake_home):
        clean_env.setenv("KAGGLE_API_TOKEN", FAKE_TOKEN)
        assert credential_source(home=fake_home) == "KAGGLE_API_TOKEN"

    def test_detects_access_token_file(self, clean_env, fake_home):
        token_file = fake_home / ".kaggle" / "access_token"
        token_file.parent.mkdir(parents=True)
        token_file.write_text(FAKE_TOKEN, encoding="utf-8")
        assert credential_source(home=fake_home) == "~/.kaggle/access_token"

    def test_blank_api_token_is_not_credentials(self, clean_env, fake_home):
        clean_env.setenv("KAGGLE_API_TOKEN", "   ")
        assert credential_source(home=fake_home) is None

    def test_empty_access_token_file_is_not_credentials(self, clean_env, fake_home):
        token_file = fake_home / ".kaggle" / "access_token"
        token_file.parent.mkdir(parents=True)
        token_file.write_text("", encoding="utf-8")
        assert credential_source(home=fake_home) is None


class TestLegacyAuth:
    def test_detects_username_and_key_pair(self, clean_env, fake_home):
        clean_env.setenv("KAGGLE_USERNAME", "someone")
        clean_env.setenv("KAGGLE_KEY", FAKE_TOKEN)
        assert credential_source(home=fake_home) == "KAGGLE_USERNAME/KAGGLE_KEY"

    def test_username_without_key_is_not_credentials(self, clean_env, fake_home):
        clean_env.setenv("KAGGLE_USERNAME", "someone")
        assert credential_source(home=fake_home) is None

    def test_key_without_username_is_not_credentials(self, clean_env, fake_home):
        clean_env.setenv("KAGGLE_KEY", FAKE_TOKEN)
        assert credential_source(home=fake_home) is None

    def test_detects_kaggle_json(self, clean_env, fake_home):
        config = fake_home / ".kaggle" / "kaggle.json"
        config.parent.mkdir(parents=True)
        config.write_text(json.dumps({"username": "someone", "key": FAKE_TOKEN}), encoding="utf-8")
        assert credential_source(home=fake_home) == "~/.kaggle/kaggle.json"


class TestNoCredentials:
    def test_returns_none_when_nothing_is_configured(self, clean_env, fake_home):
        assert credential_source(home=fake_home) is None

    def test_returns_none_when_kaggle_dir_exists_but_is_empty(self, clean_env, fake_home):
        (fake_home / ".kaggle").mkdir()
        assert credential_source(home=fake_home) is None


class TestConfigDirOverride:
    def test_honours_kaggle_config_dir(self, clean_env, tmp_path):
        """Kaggle lets users relocate the config directory; detection must follow it."""
        config_dir = tmp_path / "elsewhere"
        config_dir.mkdir()
        (config_dir / "kaggle.json").write_text(
            json.dumps({"username": "someone", "key": FAKE_TOKEN}), encoding="utf-8"
        )
        clean_env.setenv("KAGGLE_CONFIG_DIR", str(config_dir))
        assert credential_source(home=tmp_path / "unused-home") == "~/.kaggle/kaggle.json"


class TestPrecedence:
    def test_env_token_wins_over_files(self, clean_env, fake_home):
        clean_env.setenv("KAGGLE_API_TOKEN", FAKE_TOKEN)
        config = fake_home / ".kaggle" / "kaggle.json"
        config.parent.mkdir(parents=True)
        config.write_text(json.dumps({"username": "a", "key": "b"}), encoding="utf-8")
        assert credential_source(home=fake_home) == "KAGGLE_API_TOKEN"


class TestNeverLeaksSecrets:
    """The return value is displayed to the user, so it must name the method, not the secret."""

    def test_api_token_value_is_never_returned(self, clean_env, fake_home):
        clean_env.setenv("KAGGLE_API_TOKEN", FAKE_TOKEN)
        assert FAKE_TOKEN not in (credential_source(home=fake_home) or "")

    def test_key_value_is_never_returned(self, clean_env, fake_home):
        clean_env.setenv("KAGGLE_USERNAME", "someone")
        clean_env.setenv("KAGGLE_KEY", FAKE_TOKEN)
        assert FAKE_TOKEN not in (credential_source(home=fake_home) or "")

    def test_kaggle_json_contents_are_never_read_into_the_result(self, clean_env, fake_home):
        config = fake_home / ".kaggle" / "kaggle.json"
        config.parent.mkdir(parents=True)
        config.write_text(json.dumps({"username": "someone", "key": FAKE_TOKEN}), encoding="utf-8")
        assert FAKE_TOKEN not in (credential_source(home=fake_home) or "")


class TestOAuthCredentials:
    """`kaggle auth login` is the mechanism Kaggle now recommends first.

    It caches to ~/.kaggle/credentials.json with no token for the user to manage, so a
    detector that ignores it reports "no credentials" on a machine that is fully logged in.
    """

    def test_detects_oauth_credentials_json(self, clean_env, fake_home):
        creds = fake_home / ".kaggle" / "credentials.json"
        creds.parent.mkdir(parents=True)
        creds.write_text(json.dumps({"refresh_token": FAKE_TOKEN}), encoding="utf-8")
        assert credential_source(home=fake_home) == "~/.kaggle/credentials.json"

    def test_oauth_token_value_is_never_returned(self, clean_env, fake_home):
        creds = fake_home / ".kaggle" / "credentials.json"
        creds.parent.mkdir(parents=True)
        creds.write_text(json.dumps({"refresh_token": FAKE_TOKEN}), encoding="utf-8")
        assert FAKE_TOKEN not in (credential_source(home=fake_home) or "")


class TestWindowsTokenFile:
    """Kaggle explicitly tolerates a .txt suffix because Windows adds one silently."""

    def test_detects_access_token_txt(self, clean_env, fake_home):
        token_file = fake_home / ".kaggle" / "access_token.txt"
        token_file.parent.mkdir(parents=True)
        token_file.write_text(FAKE_TOKEN, encoding="utf-8")
        assert credential_source(home=fake_home) == "~/.kaggle/access_token.txt"


class TestXdgFallback:
    """On Linux with no ~/.kaggle, Kaggle follows the XDG base directory spec."""

    def test_uses_xdg_config_home_when_kaggle_dir_absent(self, clean_env, tmp_path, monkeypatch):
        monkeypatch.setattr("sys.platform", "linux")
        xdg = tmp_path / "xdg"
        (xdg / "kaggle").mkdir(parents=True)
        (xdg / "kaggle" / "kaggle.json").write_text(
            json.dumps({"username": "someone", "key": FAKE_TOKEN}), encoding="utf-8"
        )
        monkeypatch.setenv("XDG_CONFIG_HOME", str(xdg))
        assert credential_source(home=tmp_path / "home-without-kaggle") == "~/.kaggle/kaggle.json"

    def test_existing_kaggle_dir_takes_precedence_over_xdg(self, clean_env, tmp_path, monkeypatch):
        monkeypatch.setattr("sys.platform", "linux")
        home = tmp_path / "home"
        (home / ".kaggle").mkdir(parents=True)
        (home / ".kaggle" / "kaggle.json").write_text(
            json.dumps({"username": "someone", "key": FAKE_TOKEN}), encoding="utf-8"
        )
        monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
        assert credential_source(home=home) == "~/.kaggle/kaggle.json"
