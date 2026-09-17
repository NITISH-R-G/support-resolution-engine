"""Verify a rewritten history against the pre-rewrite backup, before anything is pushed.

Checks (every one reported, none assumed):
1. Same number of commits, same order; every commit's author, committer, dates and parent
   structure preserved; messages identical except the listed text replacements.
2. The old -> new commit map (filter-repo's commit-map), written as a table.
3. The rewritten tip's tree equals the pre-rewrite remediation commit's tree (the current,
   already text-free tree must not change).
4. Per commit: which paths changed, so the rewrite's blast radius is explicit.
5. Every blob reachable from any ref in the rewritten repository: 0 detector matches
   (scripts/audit/tweet_text.py) and 0 text fields in data/ and reports/ JSON; every commit
   message: 0 detector matches.
Negative control: the same scan over the backup must find tweet text, or check 5 proves nothing.

Usage:
    python scripts/audit/verify_rewritten_history.py --old BACKUP.git --new REWRITTEN.git \
        --tip remediation --out report.json
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "audit"))
sys.path.insert(0, str(ROOT / "tests"))

from tweet_text import TweetIndex, scan_bytes  # noqa: E402
from test_no_tweet_text import _violations  # noqa: E402

# The reworded commit-message quotes are dataset text, so they are read from the same gitignored
# local file the rewrite used (data/local/rewrite_replacements.json), never stored here.
def message_replacements() -> list[tuple[str, str]]:
    path = ROOT / "data" / "local" / "rewrite_replacements.json"
    if not path.exists():
        raise SystemExit(f"{path} is required (local only)")
    return [tuple(pair) for pair in json.loads(path.read_text(encoding="utf-8"))["message_replacements"]]


QUOTED = re.compile(r"'([^'\n]{12,})'|" + r'"([^"\n]{12,})"')


def git(repo: Path, *args: str, binary: bool = False, stdin: bytes | None = None):
    out = subprocess.run(["git", "-C", str(repo), *args], capture_output=True, check=True, input=stdin).stdout
    return out if binary else out.decode("utf-8")


def commit_meta(repo: Path, sha: str) -> dict:
    fmt = "%an%x00%ae%x00%aI%x00%cn%x00%ce%x00%cI%x00%P%x00%B"
    parts = git(repo, "log", "-1", f"--format={fmt}", sha).split("\x00")
    return {"author": parts[0], "author_email": parts[1], "author_date": parts[2], "committer": parts[3],
            "committer_email": parts[4], "committer_date": parts[5], "parents": parts[6].split(),
            "message": parts[7]}


def reachable_blobs(repo: Path) -> dict[str, set[str]]:
    blobs: dict[str, set[str]] = defaultdict(set)
    objects = git(repo, "rev-list", "--all", "--objects").splitlines()
    shas = [line.split(" ", 1) for line in objects]
    kinds = git(repo, "cat-file", "--batch-check=%(objectname) %(objecttype)",
                stdin="\n".join(s[0] for s in shas).encode()).splitlines()
    kind = dict(line.split() for line in kinds)
    for parts in shas:
        if len(parts) == 2 and kind.get(parts[0]) == "blob":
            blobs[parts[0]].add(parts[1])
    return blobs


def scan_repo(repo: Path, index: TweetIndex) -> dict:
    report = {"blobs": 0, "blobs_with_tweet_text": [], "blobs_with_text_fields": [], "messages_with_tweet_text": []}
    for sha, paths in reachable_blobs(repo).items():
        report["blobs"] += 1
        data = git(repo, "cat-file", "blob", sha, binary=True)
        for path in sorted(paths):
            matches, _ = scan_bytes(path, data, index)
            if matches:
                report["blobs_with_tweet_text"].append({"blob": sha[:12], "path": path, "matches": len(matches)})
            if path.startswith(("data/", "reports/")) and path.endswith((".json", ".jsonl")):
                try:
                    text = data.decode("utf-8")
                    found = []
                    if path.endswith(".jsonl"):
                        for n, line in enumerate(text.splitlines(), 1):
                            if line.strip():
                                found += _violations(json.loads(line), f"line{n}")
                    else:
                        found = _violations(json.loads(text))
                except (UnicodeDecodeError, json.JSONDecodeError):
                    found = ["unparseable"]
                if found:
                    report["blobs_with_text_fields"].append({"blob": sha[:12], "path": path, "fields": len(found)})
    for sha in git(repo, "rev-list", "--all").split():
        message = commit_meta(repo, sha)["message"]
        # Whole lines and quoted spans: a short quote inside a longer sentence is not a whole line.
        spans = message.splitlines() + [s for pair in QUOTED.findall(message) for s in pair if s]
        hits = [span for span in spans if index.match(span)]
        if hits:
            report["messages_with_tweet_text"].append({"commit": sha[:12], "lines": len(hits)})
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old", type=Path, required=True)
    parser.add_argument("--new", type=Path, required=True)
    parser.add_argument("--tip", default="remediation")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    commit_map = {}
    for line in (args.new / "filter-repo" / "commit-map").read_text().splitlines()[1:]:
        old, new = line.split()
        commit_map[old] = new
    old_commits = git(args.old, "rev-list", "--reverse", args.tip).split()
    new_commits = git(args.new, "rev-list", "--reverse", args.tip).split()

    metadata_problems, changed_paths = [], {}
    for old in old_commits:
        new = commit_map.get(old)
        if new is None:
            metadata_problems.append(f"{old[:7]}: not in commit map")
            continue
        a, b = commit_meta(args.old, old), commit_meta(args.new, new)
        expected_message = a["message"]
        for before, after in message_replacements():
            expected_message = expected_message.replace(before, after)
        # filter-repo rewrites abbreviated commit ids quoted in messages to the new ids; accept
        # exactly that substitution and nothing else.
        for old_full, new_full in commit_map.items():
            for length in (7, 8, 10, 12, 40):
                expected_message = re.sub(rf"\b{old_full[:length]}\b", new_full[:length], expected_message)
        for field in ("author", "author_email", "author_date", "committer", "committer_email", "committer_date"):
            if a[field] != b[field]:
                metadata_problems.append(f"{old[:7]}: {field} differs")
        if [commit_map.get(p) for p in a["parents"]] != b["parents"]:
            metadata_problems.append(f"{old[:7]}: parents differ")
        if expected_message != b["message"]:
            import difflib

            diff = [line for line in difflib.unified_diff(expected_message.splitlines(), b["message"].splitlines(),
                                                          lineterm="", n=0) if line[:1] in "+-" and line[:3] not in ("+++", "---")]
            metadata_problems.append(f"{old[:7]}: message differs beyond the listed replacements: {diff[:4]}")
        old_files = dict(line.split("\t", 1)[::-1] for line in git(args.old, "ls-tree", "-r", "--format=%(objectname)\t%(path)", old).splitlines())
        new_files = dict(line.split("\t", 1)[::-1] for line in git(args.new, "ls-tree", "-r", "--format=%(objectname)\t%(path)", new).splitlines())
        if set(old_files) != set(new_files):
            metadata_problems.append(f"{old[:7]}: file set differs")
        changed = sorted(p for p in old_files if old_files[p] != new_files.get(p))
        if changed:
            changed_paths[old[:7]] = changed

    old_tree = git(args.old, "rev-parse", f"{args.tip}^{{tree}}").strip()
    new_tree = git(args.new, "rev-parse", f"{args.tip}^{{tree}}").strip()

    index = TweetIndex()
    new_scan = scan_repo(args.new, index)
    old_scan = scan_repo(args.old, index)

    report = {
        "commits_old": len(old_commits),
        "commits_new": len(new_commits),
        "metadata_problems": metadata_problems,
        "tip_tree_identical": old_tree == new_tree,
        "commits_with_changed_paths": len(changed_paths),
        "changed_paths_by_commit": changed_paths,
        "commit_map": [{"old": o[:7], "new": commit_map[o][:7], "changed": o in [k for k in commit_map] and commit_map[o] != o}
                       for o in old_commits],
        "rewritten_scan": new_scan,
        "negative_control_backup_scan": {
            "blobs": old_scan["blobs"],
            "blobs_with_tweet_text": len(old_scan["blobs_with_tweet_text"]),
            "blobs_with_text_fields": len(old_scan["blobs_with_text_fields"]),
            "messages_with_tweet_text": len(old_scan["messages_with_tweet_text"]),
        },
    }
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    summary = {k: v for k, v in report.items() if k not in ("changed_paths_by_commit", "commit_map", "rewritten_scan")}
    summary["rewritten_scan"] = {k: (len(v) if isinstance(v, list) else v) for k, v in new_scan.items()}
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
