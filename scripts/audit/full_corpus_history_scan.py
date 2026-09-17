"""Scan every text blob in a repository's history against the FULL corpus (all 108 brands).

The main detector (tweet_text.py) indexes AppleSupport tweets only, but the project analysed
every brand during brand selection, so other brands' tweets could have been quoted too. This
scans every reachable blob, extracting:
    Markdown / text   quoted spans and blockquote lines
    Python            string literals (adjacent literals merged)
    JSON / JSONL      string values
It keeps spans of >= 7 words and matches each against all 2.8M tweets. A candidate tweet must
contain the span's two rarest words (via an inverted index of words occurring in < 5,000
tweets). A match is sequence similarity >= 0.8, or a common run covering >= 80% of the span.

Output: counts, paths, blob ids and tweet ids only. No text.

Usage:
    python scripts/audit/full_corpus_history_scan.py --repo PATH [--out report.json]
"""
from __future__ import annotations

import argparse
import ast
import difflib
import html
import io
import json
import re
import subprocess
import sys
import tokenize
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
WORD = re.compile(r"[a-z0-9']+")
QUOTE = re.compile(r'"([^"\n]{30,})"|“([^”\n]{30,})”|\*([^*\n]{30,})\*|^>\s*(.{30,})$', re.M)
MARKER = re.compile(r"\[(tweet-)?text redacted: [^\]]*\]")
RARE_LIMIT = 5000


def norm(text: str) -> str:
    text = re.sub(r"@\w+|https?://\S+", " ", html.unescape(str(text)).lower())
    return " ".join(WORD.findall(text))


def spans_of(path: str, data: bytes) -> set[str]:
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        return set()
    out: set[str] = set()
    if path.endswith((".json", ".jsonl")):
        def walk(v):
            if isinstance(v, dict):
                for x in v.values():
                    walk(x)
            elif isinstance(v, list):
                for x in v:
                    walk(x)
            elif isinstance(v, str):
                out.add(v)
        docs = text.splitlines() if path.endswith(".jsonl") else [text]
        for doc in docs:
            try:
                walk(json.loads(doc))
            except json.JSONDecodeError:
                pass
    elif path.endswith(".py"):
        try:
            group = []
            for tok in tokenize.generate_tokens(io.StringIO(text).readline):
                if tok.type == tokenize.STRING:
                    group.append(tok.string)
                elif tok.type in (tokenize.NL, tokenize.NEWLINE, tokenize.COMMENT) and group:
                    continue
                else:
                    if group:
                        try:
                            value = "".join(ast.literal_eval(s) for s in group)
                            if isinstance(value, str):
                                out.add(value)
                        except (ValueError, SyntaxError, TypeError):
                            pass
                    group = []
        except (tokenize.TokenError, SyntaxError):
            pass
    else:
        for m in QUOTE.finditer(text):
            out.add(next(g for g in m.groups() if g))
    return {s for s in out if not MARKER.search(s) and len(norm(s).split()) >= 7}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    import pandas as pd

    objects = subprocess.run(["git", "-C", str(args.repo), "rev-list", "--all", "--objects"],
                             capture_output=True, text=True, check=True).stdout.splitlines()
    paths = defaultdict(set)
    for line in objects:
        sha, _, path = line.partition(" ")
        if path and path.endswith((".md", ".txt", ".py", ".json", ".jsonl", ".toml", ".csv")):
            paths[sha].add(path)
    span_locations: dict[str, set[str]] = defaultdict(set)
    for sha, ps in paths.items():
        data = subprocess.run(["git", "-C", str(args.repo), "cat-file", "blob", sha], capture_output=True).stdout
        for path in ps:
            for span in spans_of(path, data):
                span_locations[norm(span)].add(f"{path}@{sha[:10]}")
    print("distinct spans:", len(span_locations), file=sys.stderr)

    frame = pd.read_csv(ROOT / "data" / "raw" / "twcs" / "twcs.csv", usecols=["tweet_id", "author_id", "text"], dtype=str)
    texts = [norm(t) for t in frame["text"].values]
    df = Counter()
    for t in texts:
        df.update(set(t.split()))
    postings: dict[str, list[int]] = defaultdict(list)
    for i, t in enumerate(texts):
        for w in set(t.split()):
            if df[w] < RARE_LIMIT and len(w) >= 4:
                postings[w].append(i)

    hits = []
    for span, locations in span_locations.items():
        words = sorted({w for w in span.split() if w in postings}, key=lambda w: df[w])
        if len(words) < 2:
            continue
        candidates = set(postings[words[0]]) & set(postings[words[1]])
        best = (0.0, None)
        for i in candidates:
            matcher = difflib.SequenceMatcher(None, span, texts[i], autojunk=False)
            ratio = matcher.ratio()
            longest = matcher.find_longest_match(0, len(span), 0, len(texts[i])).size / max(len(span), 1)
            score = max(ratio, longest)
            if score > best[0]:
                best = (score, i)
        if best[0] >= 0.8:
            hits.append({"score": round(best[0], 3), "words": len(span.split()),
                         "tweet_id": frame["tweet_id"].iat[best[1]], "author": frame["author_id"].iat[best[1]],
                         "locations": sorted(locations)[:6]})
    report = {"blobs_scanned": len(paths), "distinct_spans": len(span_locations), "hits": len(hits),
              "details": sorted(hits, key=lambda h: -h["score"])}
    if args.out:
        args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "details"}, indent=2))
    for h in report["details"][:40]:
        print(h)


if __name__ == "__main__":
    main()
