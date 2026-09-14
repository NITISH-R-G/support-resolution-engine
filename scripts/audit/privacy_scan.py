"""Audit F: where does real customer text live in the repository and its git history?

Customer and brand tweets from the AppleSupport corpus (data/interim, gitignored) are cut into
overlapping 8-word shingles. Every tracked file, and every blob reachable from any ref in git
history, is shingled the same way. A corpus message counts as REPRODUCED in a file when at least
3 of its shingles occur there (all of them, for a message with fewer than 3). Output is counts
and paths only: this script never prints or stores any text.

Also counts PII-shaped strings (email addresses, phone-like numbers) in the same files.

Limitation, stated rather than hidden: messages under 8 words produce no shingle and cannot be
detected this way. Paraphrases are not detected either.

Negative control: a file planted with one known customer message must be detected, and a file
of invented text must not.

Usage:
    python scripts/audit/privacy_scan.py      (needs data/interim from fetch_data + discovery)
"""
from __future__ import annotations

import json
import pickle
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
PAIRS = ROOT / "data" / "interim" / "applesupport_pairs_limit_all.pkl"
OUT = ROOT / "reports" / "privacy_scan.json"

WORD = re.compile(r"[a-z0-9']+")
N = 8
MIN_SHINGLES = 3
EMAIL = re.compile(rb"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
# Phone-shaped: 10-15 digits written with at least one phone separator (space, dash, parens or
# a leading +). Bare digit runs are excluded: pair ids, timestamps and metrics dominate those.
PHONE = re.compile(rb"(?<![\w.])(?=[\d\s().+-]*[\s()+-])\+?\(?\d{2,4}\)?(?:[\s.-]?\d{2,4}){2,4}(?![\w.])")


def _phone_like(data: bytes) -> int:
    return sum(1 for m in PHONE.finditer(data)
               if 10 <= sum(ch.isdigit() for ch in m.group().decode()) <= 15
               and re.search(rb"[\s()+-]", m.group()))
PLACEHOLDER_EMAILS = (b"example.com", b"anthropic.com", b"users.noreply.github.com")


def shingles(text: str) -> set[int]:
    words = WORD.findall(text.lower())
    return {hash(tuple(words[i:i + N])) for i in range(len(words) - N + 1)}


def build_index():
    with PAIRS.open("rb") as handle:
        pairs = pickle.load(handle)
    index = {"customer": defaultdict(set), "brand": defaultdict(set)}
    sizes = {"customer": {}, "brand": {}}
    seen = set()
    for pair in pairs:
        tweets = [(pair.customer_tweet, "customer"), (pair.support_tweet, "brand")]
        tweets += [(t, "customer" if t.inbound else "brand") for t in pair.context]
        for tweet, kind in tweets:
            if tweet.tweet_id in seen:
                continue
            seen.add(tweet.tweet_id)
            s = shingles(tweet.text)
            if not s:
                continue
            sizes[kind][tweet.tweet_id] = len(s)
            for h in s:
                index[kind][h].add(tweet.tweet_id)
    return index, sizes, len(seen)


def reproduced(data: bytes, index, sizes) -> dict:
    s = shingles(data.decode("utf-8", errors="ignore"))
    result = {}
    for kind in ("customer", "brand"):
        hits = defaultdict(int)
        for h in s:
            for tid in index[kind].get(h, ()):
                hits[tid] += 1
        result[kind] = sum(1 for tid, n in hits.items() if n >= min(MIN_SHINGLES, sizes[kind][tid]))
    emails = [m for m in EMAIL.findall(data) if not any(p in m for p in PLACEHOLDER_EMAILS)]
    result["email_like"] = len(emails)
    result["phone_like"] = _phone_like(data)
    return result


def git(*args, input_bytes=None) -> bytes:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, check=True,
                          input=input_bytes).stdout


def main() -> None:
    index, sizes, n_tweets = build_index()

    # Negative controls.
    with PAIRS.open("rb") as handle:
        pairs = pickle.load(handle)
    known_text = next(p.customer_text for p in pairs if len(shingles(p.customer_text)) >= 6)
    planted = reproduced(("notes\n" + known_text + "\nend").encode(), index, sizes)["customer"] >= 1
    invented = reproduced(b"the quick brown fox jumps over the lazy dog while seven purple "
                          b"elephants recite algebra quietly under a silver moon", index, sizes)
    planted_pii = reproduced(b"reach me at jane.roe@mailbox.org or +1 415-555-0134", index, sizes)
    ids_only = reproduced(b'{"pair_id": "2690875__2690877", "latency_ms": 12345678901, "x": 0.4166666666666667}',
                          index, sizes)
    controls = {"planted_customer_message_detected": planted,
                "invented_text_not_detected": invented["customer"] == 0 and invented["brand"] == 0,
                "planted_email_and_phone_detected": planted_pii["email_like"] == 1 and planted_pii["phone_like"] == 1,
                "ids_and_numbers_not_counted_as_phones": ids_only["phone_like"] == 0}
    del pairs, known_text

    tracked = {}
    for path in git("ls-files").decode().splitlines():
        file = ROOT / path
        if file.is_file():
            r = reproduced(file.read_bytes(), index, sizes)
            if any(r.values()):
                tracked[path] = r

    # History: every blob reachable from any ref, with the paths it was stored under.
    objects = git("rev-list", "--all", "--objects").decode().splitlines()
    on_main = {line.split(" ", 1)[0] for line in git("rev-list", "main", "--objects").decode().splitlines()}
    head_blobs = {line.split()[2] for line in git("ls-tree", "-r", "HEAD").decode().splitlines()}
    paths = defaultdict(set)
    for line in objects:
        sha, _, path = line.partition(" ")
        if path:
            paths[sha].add(path)
    kinds = git("cat-file", "--batch-check=%(objectname) %(objecttype)",
                input_bytes="\n".join(paths).encode()).decode().split("\n")
    blobs = [line.split()[0] for line in kinds if line.endswith(" blob")]
    history = []
    for sha in blobs:
        r = reproduced(git("cat-file", "blob", sha), index, sizes)
        if r["customer"] or r["email_like"] or r["phone_like"]:
            history.append({"blob": sha, "paths": sorted(paths[sha]), "reachable_from_main": sha in on_main,
                            "in_head_tree": sha in head_blobs, **r})

    only_history = [h for h in history if not h["in_head_tree"] and h["customer"]]
    report = {
        "corpus_tweets_indexed": n_tweets,
        "method": f"{N}-word shingles, >= {MIN_SHINGLES} matching per message; counts only",
        "negative_controls": controls,
        "tracked_files_with_matches": tracked,
        "history_blobs_scanned": len(blobs),
        "history_blobs_with_customer_text_or_pii_shapes": len(history),
        "history_blobs_with_customer_text_not_in_head": [
            {k: h[k] for k in ("blob", "paths", "reachable_from_main", "customer")} for h in only_history],
        "history_blobs": history,
    }
    OUT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"controls": controls, "tracked_files_with_matches": len(tracked),
                      "history_blobs_scanned": len(blobs),
                      "history_blobs_with_customer_text_not_in_head": len(only_history)}, indent=2))
    for path, r in sorted(tracked.items(), key=lambda kv: -kv[1]["customer"]):
        print(f"  {r['customer']:>5} customer  {r['brand']:>5} brand  {r['email_like']:>3} email  "
              f"{r['phone_like']:>3} phone  {path}")


if __name__ == "__main__":
    main()
