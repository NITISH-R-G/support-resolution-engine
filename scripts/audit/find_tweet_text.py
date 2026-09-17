"""Report verbatim corpus tweet text in tracked files (counts, locations and tweet ids only).

Usage:
    python scripts/audit/find_tweet_text.py                # tracked files at HEAD / working tree
    python scripts/audit/find_tweet_text.py --out report.json

Negative controls run first: a planted customer message, a planted short whole message and a
planted edited excerpt must be detected; generic brand boilerplate and invented text must not.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "audit"))

from tweet_text import TweetIndex, scan_bytes, words_of  # noqa: E402


def controls(index: TweetIndex) -> dict:
    import pickle

    with (ROOT / "data" / "interim" / "applesupport_pairs_limit_all.pkl").open("rb") as handle:
        pairs = pickle.load(handle)
    long_text = next(p.customer_text for p in pairs if len(words_of(p.customer_text)) >= 14
                     and index.match(p.customer_text))
    short_text = next(p.customer_text for p in pairs if 2 <= len(words_of(p.customer_text)) <= 4
                      and len(" ".join(words_of(p.customer_text))) >= 12 and index.match(p.customer_text))
    words = long_text.split()
    edited = " ".join(words[: len(words) // 2] + words[len(words) // 2 + 1:])  # one word deleted
    doc = json.dumps({"a": long_text, "b": {"c": [short_text]}, "d": edited}).encode()
    found = scan_bytes("x.json", doc, index)[0]
    boiler = scan_bytes("x.md", b'We said "please send us a DM with your country and we will help" today.\n', index)[0]
    invented = scan_bytes("x.py", b'X = "seven purple elephants recite algebra quietly under a silver moon"\n', index)[0]
    generic_test_phrase = scan_bytes("x.py", b'assert f("my battery drains so fast")\n', index)[0]
    planted_py = scan_bytes("x.py", ("X = " + repr(long_text) + "\n").encode(), index)[0]
    return {
        "planted_long_message_detected": any(m.location == "$.a" for m in found),
        "planted_short_whole_message_detected": any(m.location == "$.b.c[0]" for m in found),
        "planted_edited_excerpt_detected": any(m.location == "$.d" for m in found),
        "brand_boilerplate_not_flagged": not boiler,
        "invented_text_not_flagged": not invented,
        "generic_test_phrase_not_flagged": not generic_test_phrase,
        "planted_message_in_python_literal_detected": bool(planted_py),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    index = TweetIndex()
    result = {"corpus_tweets_indexed": index.tweet_count, "controls": controls(index)}
    files = subprocess.run(["git", "ls-files"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.split("\n")
    per_file = {}
    for rel in filter(None, files):
        path = ROOT / rel
        if not path.is_file():
            continue
        matches, _ = scan_bytes(rel, path.read_bytes(), index)
        if matches:
            per_file[rel] = {
                "matches": len(matches),
                "distinct_tweets": len({m.tweet_id for m in matches}),
                "rules": dict(Counter(m.rule for m in matches)),
                "locations": [m.location for m in matches][:50],
            }
    result["files_with_tweet_text"] = len(per_file)
    result["total_matches"] = sum(v["matches"] for v in per_file.values())
    result["per_file"] = per_file
    print(json.dumps({k: v for k, v in result.items() if k != "per_file"}, indent=2))
    for rel, info in sorted(per_file.items(), key=lambda kv: -kv[1]["matches"]):
        print(f"  {info['matches']:>6}  {info['distinct_tweets']:>5} tweets  {info['rules']}  {rel}")
    if args.out:
        args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
