"""Download the Customer Support on Twitter corpus.

The raw file is ~500 MB and is deliberately never committed. This script fetches it into
``data/raw/`` (git-ignored) and verifies it, so a reviewer reproduces the dataset rather
than downloading a copy of it from us.

Credentials — any one of the four mechanisms Kaggle supports:
  * ``KAGGLE_API_TOKEN`` environment variable
  * ``KAGGLE_USERNAME`` + ``KAGGLE_KEY`` environment variables
  * ``~/.kaggle/kaggle.json`` from https://www.kaggle.com/settings -> "Create New Token"
  * ``~/.kaggle/access_token``

``KAGGLE_CONFIG_DIR`` relocates the config directory and is honoured. Detection lives in
``hiver_support.kaggle_auth`` and reports only *which* mechanism is present, never the
credential value, so nothing secret reaches the terminal or a log.

Usage:
    python scripts/fetch_data.py            # download + verify
    python scripts/fetch_data.py --check    # verify an existing copy only
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hiver_support.kaggle_auth import (  # noqa: E402
    credential_source,
    missing_credentials_message,
)

DATASET = "thoughtvector/customer-support-on-twitter"
CSV_NAME = "twcs/twcs.csv"
ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"

EXPECTED_COLUMNS = {
    "tweet_id",
    "author_id",
    "inbound",
    "created_at",
    "text",
    "response_tweet_id",
    "in_response_to_tweet_id",
}


def locate_csv() -> Path | None:
    for candidate in (RAW_DIR / "twcs.csv", RAW_DIR / CSV_NAME):
        if candidate.exists():
            return candidate
    return None


def download() -> Path:
    source = credential_source()
    if source is None:
        sys.exit(missing_credentials_message())
    # Names the mechanism only; the credential value is never read or printed.
    print(f"Authenticating with Kaggle via {source}")

    # Imported lazily: the package authenticates at import time and exits if creds are absent.
    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {DATASET} into {RAW_DIR} (~500 MB, one time)...")
    api.dataset_download_files(DATASET, path=str(RAW_DIR), unzip=True, quiet=False)

    csv_path = locate_csv()
    if csv_path is None:
        sys.exit(f"Download finished but no CSV was found under {RAW_DIR}")
    return csv_path


def verify(csv_path: Path) -> None:
    """Confirm the file really is the corpus we wrote the pipeline against."""
    import pandas as pd

    head = pd.read_csv(csv_path, nrows=5)
    missing = EXPECTED_COLUMNS - set(head.columns)
    if missing:
        sys.exit(f"Unexpected schema in {csv_path}; missing columns: {sorted(missing)}")

    size_mb = csv_path.stat().st_size / 1024**2
    print(f"OK  {csv_path}  ({size_mb:.0f} MB)")
    print(f"    columns: {list(head.columns)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="verify an existing copy, do not download")
    args = parser.parse_args()

    csv_path = locate_csv()
    if csv_path is None:
        if args.check:
            sys.exit(f"No corpus found under {RAW_DIR}. Run without --check to download it.")
        csv_path = download()
    else:
        print(f"Found existing corpus at {csv_path}")

    verify(csv_path)


if __name__ == "__main__":
    main()
