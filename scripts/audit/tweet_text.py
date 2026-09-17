"""Find (and optionally redact) verbatim corpus tweet text inside repository files.

The earlier 8-word shingle scan (privacy_scan.py) could not see short quotes or edited excerpts.
This detector works structurally instead: it extracts candidate strings from each file, then
matches every candidate against an index of all AppleSupport tweets (customer and brand, thread
context included).

Candidates:
    .json / .jsonl   every string value, with its JSON path
    .py              every string literal (adjacent literals merged, as Python concatenates them)
    other text       quoted spans ("...", curly quotes, *...*, `...`) and every whole line

Match rules, on a normalised form (HTML-unescaped, lowercased, handles and URLs removed, words
only):
    SHORT  2-4 words and >= 12 characters: equals an entire tweet that occurs at most 3 times
           in the corpus, so a generic "thank you so much" is never flagged.
    LONG   >= 5 words: at least half of the candidate's 5-word shingles are *rare* shingles
           (occurring in <= 3 tweets) belonging to one and the same tweet. Brand boilerplate such
           as "please send us a dm" is common and therefore never decisive.

A match records the tweet id and a sha256 of the matched original string, never the text.

Redaction replaces the matched string with
    [tweet-text redacted: tweet_id=<id> sha256=<first 16 hex>]
The id lets a reviewer with the Kaggle download find the tweet; the hash lets them confirm it.

This module is shared by the privacy audit, the tree redaction and the history rewrite, so all
three apply exactly the same rule.
"""
from __future__ import annotations

import ast
import hashlib
import html
import io
import json
import pickle
import re
import tokenize
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
import sys  # noqa: E402

sys.path.insert(0, str(ROOT / "src"))  # the pickled pairs reference hiver_support classes
PAIRS = ROOT / "data" / "interim" / "applesupport_pairs_limit_all.pkl"

_HANDLE = re.compile(r"@\w+")
_URL = re.compile(r"https?://\S+")
_WORD = re.compile(r"[a-z0-9']+")
_QUOTED = re.compile(r'"([^"\n]{12,})"|“([^”\n]{12,})”|\*"?([^*\n]{12,}?)"?\*|`([^`\n]{12,})`')
MAX_DF = 3
N = 5
MARKER_RE = re.compile(r"\[tweet-text redacted: tweet_id=\d+ sha256=[0-9a-f]{16}\]")


def words_of(text: str) -> list[str]:
    text = html.unescape(text).lower()
    text = _URL.sub(" ", _HANDLE.sub(" ", text))
    return _WORD.findall(text)


def marker(tweet_id: str, original: str) -> str:
    digest = hashlib.sha256(original.encode("utf-8")).hexdigest()[:16]
    return f"[tweet-text redacted: tweet_id={tweet_id} sha256={digest}]"


@dataclass(frozen=True)
class Match:
    location: str
    tweet_id: str
    rule: str
    sha256: str


class TweetIndex:
    def __init__(self, pairs_path: Path = PAIRS) -> None:
        with pairs_path.open("rb") as handle:
            pairs = pickle.load(handle)
        self.ids: list[str] = []
        self.lengths: list[int] = []
        whole: dict[str, list[int]] = defaultdict(list)
        shingle_first: dict[int, int] = {}
        shingle_df: Counter = Counter()
        seen = set()
        for pair in pairs:
            for tweet in (pair.customer_tweet, pair.support_tweet, *pair.context):
                if tweet.tweet_id in seen:
                    continue
                seen.add(tweet.tweet_id)
                index = len(self.ids)
                self.ids.append(tweet.tweet_id)
                words = words_of(tweet.text)
                self.lengths.append(len(words))
                if len(words) >= 2:
                    whole[" ".join(words)].append(index)
                for h in {hash(tuple(words[i:i + N])) for i in range(len(words) - N + 1)}:
                    shingle_df[h] += 1
                    shingle_first.setdefault(h, index)
        self.whole = {k: v for k, v in whole.items() if len(v) <= MAX_DF}
        self.rare = {h: i for h, i in shingle_first.items() if shingle_df[h] <= MAX_DF}
        self.tweet_count = len(self.ids)

    def match(self, text: str, allow_short: bool = True) -> tuple[str, str] | None:
        """Return (tweet_id, rule) when ``text`` reproduces a corpus tweet, else None.

        ``allow_short=False`` disables the whole-message rule for 2-4 word strings. Used for
        Python literals, where short strings are identifiers, enum values and phrases written
        for tests rather than quotations.
        """
        if MARKER_RE.fullmatch(text.strip()):
            return None
        words = words_of(text)
        if not words:
            return None
        joined = " ".join(words)
        if 2 <= len(words) <= 4:
            if allow_short and len(joined) >= 12 and joined in self.whole:
                return self.ids[self.whole[joined][0]], "short-whole"
            return None
        if len(words) < N:
            return None
        shingles = {hash(tuple(words[i:i + N])) for i in range(len(words) - N + 1)}
        votes = Counter(self.rare[h] for h in shingles if h in self.rare)
        if not votes:
            return None
        tweet, count = votes.most_common(1)[0]
        if count < len(shingles) / 2:
            return None
        # One shared rare shingle is not enough on its own: a short generic phrase written for a
        # test ("my battery drains so fast") can collide with one 5-gram of a long real tweet.
        # Require a second shared shingle, or that the candidate covers most of that tweet.
        if count >= 2 or len(words) >= 0.6 * self.lengths[tweet]:
            return self.ids[tweet], "long-shingles"
        return None


# ----------------------------------------------------------------------------- per file type

def _json_walk(value, path, index, matches, redact):
    if isinstance(value, dict):
        return {k: _json_walk(v, f"{path}.{k}", index, matches, redact) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_walk(v, f"{path}[{i}]", index, matches, redact) for i, v in enumerate(value)]
    if isinstance(value, str):
        hit = index.match(value)
        if hit:
            matches.append(Match(path, hit[0], hit[1], hashlib.sha256(value.encode()).hexdigest()[:16]))
            if redact:
                return marker(hit[0], value)
    return value


def scan_json(data: bytes, index: TweetIndex, redact: bool, jsonl: bool):
    text = data.decode("utf-8")
    matches: list[Match] = []
    if jsonl:
        out_lines = []
        for n, line in enumerate(text.splitlines(keepends=True)):
            body = line.rstrip("\r\n")
            ending = line[len(body):]
            if not body.strip():
                out_lines.append(line)
                continue
            try:
                record = json.loads(body)
            except json.JSONDecodeError:
                out_lines.append(line)
                continue
            before = len(matches)
            new = _json_walk(record, f"line{n + 1}", index, matches, redact)
            if redact and len(matches) > before:
                out_lines.append(json.dumps(new, ensure_ascii=False) + ending)
            else:
                out_lines.append(line)
        return matches, "".join(out_lines).encode("utf-8")
    try:
        document = json.loads(text)
    except json.JSONDecodeError:
        return scan_text(data, index, redact)
    new = _json_walk(document, "$", index, matches, redact)
    if redact and matches:
        indent = 2 if "\n  " in text[:200] else None
        trailing = "\n" if text.endswith("\n") else ""
        return matches, (json.dumps(new, indent=indent, ensure_ascii=False) + trailing).encode("utf-8")
    return matches, data


def scan_python(data: bytes, index: TweetIndex, redact: bool):
    source = data.decode("utf-8")
    matches: list[Match] = []
    try:
        tokens = list(tokenize.generate_tokens(io.StringIO(source).readline))
    except (tokenize.TokenError, SyntaxError):
        return scan_text(data, index, redact)
    lines = source.splitlines(keepends=True)
    offsets = [0]
    for line in lines:
        offsets.append(offsets[-1] + len(line))

    def absolute(pos):
        return offsets[pos[0] - 1] + pos[1]

    groups, current = [], []
    for tok in tokens:
        if tok.type == tokenize.STRING:
            current.append(tok)
        elif tok.type in (tokenize.NL, tokenize.NEWLINE, tokenize.COMMENT) and current:
            continue
        else:
            if current:
                groups.append(current)
            current = []
    if current:
        groups.append(current)

    replacements = []
    for group in groups:
        try:
            value = "".join(ast.literal_eval(t.string) for t in group)
        except (ValueError, SyntaxError, TypeError):
            continue
        if not isinstance(value, str):
            continue
        hit = index.match(value, allow_short=False)
        if hit:
            matches.append(Match(f"line{group[0].start[0]}", hit[0], hit[1],
                                 hashlib.sha256(value.encode()).hexdigest()[:16]))
            replacements.append((absolute(group[0].start), absolute(group[-1].end), repr(marker(hit[0], value))))
    if redact and replacements:
        for start, end, new in sorted(replacements, reverse=True):
            source = source[:start] + new + source[end:]
        return matches, source.encode("utf-8")
    return matches, data


def scan_text(data: bytes, index: TweetIndex, redact: bool):
    text = data.decode("utf-8", errors="strict")
    matches: list[Match] = []
    out = []
    for n, line in enumerate(text.splitlines(keepends=True)):
        new_line = line
        for m in list(_QUOTED.finditer(line)):
            span = next(g for g in m.groups() if g)
            hit = index.match(span)
            if hit:
                matches.append(Match(f"line{n + 1}", hit[0], hit[1] + "-quoted",
                                     hashlib.sha256(span.encode()).hexdigest()[:16]))
                new_line = new_line.replace(span, marker(hit[0], span))
        body = new_line.rstrip("\r\n")
        hit = index.match(body)
        if hit and len(words_of(body)) >= N:
            matches.append(Match(f"line{n + 1}", hit[0], hit[1] + "-line",
                                 hashlib.sha256(body.encode()).hexdigest()[:16]))
            new_line = marker(hit[0], body) + new_line[len(body):]
        out.append(new_line)
    return matches, ("".join(out).encode("utf-8") if redact else data)


BINARY_SUFFIXES = {".png", ".pkl", ".pyc", ".jpg", ".gif", ".zip", ".whl"}


def scan_bytes(path: str, data: bytes, index: TweetIndex, redact: bool = False):
    suffix = Path(path).suffix.lower()
    if suffix in BINARY_SUFFIXES or b"\0" in data[:4096]:
        return [], data
    try:
        data.decode("utf-8")
    except UnicodeDecodeError:
        return [], data
    if suffix == ".jsonl":
        return scan_json(data, index, redact, jsonl=True)
    if suffix == ".json":
        return scan_json(data, index, redact, jsonl=False)
    if suffix == ".py":
        return scan_python(data, index, redact)
    return scan_text(data, index, redact)
