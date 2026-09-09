"""Detect which Kaggle credential mechanism is configured.

Kaggle has accumulated five authentication mechanisms. Recognising only the two legacy ones
means refusing to run on a machine that authenticates perfectly well, which is exactly the
integration bug this module was written to fix.

Detection deliberately answers *which mechanism is present*, never *what the credential is*.
The result is printed to the user and appears in error messages, so returning a token here
would put a live secret into terminal scrollback and any captured log. Credential values are
never read, returned, or written by this module; only presence and non-emptiness are checked.
Authentication itself is left entirely to the Kaggle client.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Ordered by precedence: an explicitly exported variable beats an on-disk config, which is
# what a user setting a variable for one command expects.
API_TOKEN_ENV = "KAGGLE_API_TOKEN"
USERNAME_ENV = "KAGGLE_USERNAME"
KEY_ENV = "KAGGLE_KEY"
CONFIG_DIR_ENV = "KAGGLE_CONFIG_DIR"

SUPPORTED_METHODS = (
    "kaggle auth login (OAuth, cached in ~/.kaggle/credentials.json)",
    f"{API_TOKEN_ENV} environment variable",
    f"{USERNAME_ENV} + {KEY_ENV} environment variables",
    "~/.kaggle/kaggle.json",
    "~/.kaggle/access_token (or access_token.txt)",
)


def _config_dir(home: Path) -> Path:
    """Resolve Kaggle's config directory the way the Kaggle client itself does.

    ``KAGGLE_CONFIG_DIR`` wins outright. Otherwise ``~/.kaggle`` is used when it exists;
    only on Linux, and only when it does not, does the client fall back to the XDG base
    directory. Mirroring that logic matters because a detector looking in the wrong place
    reports "no credentials" on a machine that authenticates fine.
    """
    override = os.getenv(CONFIG_DIR_ENV)
    if override:
        return Path(override)

    default = home / ".kaggle"
    if sys.platform.startswith("linux") and not default.exists():
        xdg = os.getenv("XDG_CONFIG_HOME")
        return (Path(xdg) if xdg else home / ".config") / "kaggle"
    return default


def _has_content(path: Path) -> bool:
    """True only if the file exists and is not blank.

    A zero-byte token file is a common half-finished setup, and treating it as valid
    produces a confusing authentication failure much further downstream.
    """
    try:
        return path.is_file() and bool(path.read_text(encoding="utf-8").strip())
    except OSError:
        return False


def credential_source(home: Path | None = None) -> str | None:
    """Name the Kaggle credential mechanism that is configured, if any.

    Args:
        home: Home directory to inspect. Injectable so tests never touch a developer's real
            ``~/.kaggle``.

    Returns:
        A short human-readable name of the mechanism found — one of ``SUPPORTED_METHODS`` —
        or ``None`` if no credentials are configured. **Never returns a credential value.**
    """
    home = home or Path.home()

    if (os.getenv(API_TOKEN_ENV) or "").strip():
        return API_TOKEN_ENV

    if (os.getenv(USERNAME_ENV) or "").strip() and (os.getenv(KEY_ENV) or "").strip():
        return f"{USERNAME_ENV}/{KEY_ENV}"

    config_dir = _config_dir(home)

    # OAuth first: `kaggle auth login` is the mechanism Kaggle now recommends, and a machine
    # logged in that way has no token file or environment variable to find.
    if _has_content(config_dir / "credentials.json"):
        return "~/.kaggle/credentials.json"

    if _has_content(config_dir / "kaggle.json"):
        return "~/.kaggle/kaggle.json"

    if _has_content(config_dir / "access_token"):
        return "~/.kaggle/access_token"

    # Kaggle tolerates the .txt suffix Windows appends when a file is saved from a browser
    # or editor, so a detector that ignores it fails on exactly the platform that needs it.
    if _has_content(config_dir / "access_token.txt"):
        return "~/.kaggle/access_token.txt"

    return None


def credentials_available(home: Path | None = None) -> bool:
    return credential_source(home) is not None


def missing_credentials_message(home: Path | None = None) -> str:
    """Guidance shown when nothing is configured. Contains no secret material."""
    config_dir = _config_dir(home or Path.home())
    methods = "\n".join(f"    - {method}" for method in SUPPORTED_METHODS)
    return (
        "No Kaggle credentials found. Supported methods:\n"
        f"{methods}\n\n"
        "  Easiest: run  kaggle auth login\n"
        "  Or generate a token at https://www.kaggle.com/settings/api ('Generate New Token'),\n"
        f"  then save it to {config_dir / 'access_token'}\n"
        f"  or export {API_TOKEN_ENV}."
    )
