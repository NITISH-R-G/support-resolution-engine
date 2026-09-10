"""Adjudicate contested taxonomy boundaries against real support behaviour.

Answers one question per boundary: **does support actually handle these two groups
differently?** A label earns its place by changing what support does (`SPEC.md` §5), so the
brand's own historical replies are the evidence — not vocabulary, and explicitly not
golden-set sample size, which measures our convenience rather than the operation.

Two studies:

1. **Account vs security** — four sub-groups (access, compromise, phishing, privacy) profiled
   separately, to decide whether they are one intent or several.
2. **Intent vs topic** — six contested pairs, each asked whether the distinction is topic,
   symptom, causal explanation, requested action, required resolution, or escalation
   behaviour.

Every comparison carries a bootstrap CI, and groups under 30 yield no verdict at all. Train
split only: dev, test pool and any golden candidates are never inspected.

Usage:
    python scripts/adjudicate_taxonomy.py
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from hiver_support.analysis.handling import (  # noqa: E402
    MIN_GROUP_FOR_VERDICT,
    compare_handling,
    handling_profile,
)
from hiver_support.data.normalise import normalise_text  # noqa: E402
from hiver_support.data.pii import mask_pii  # noqa: E402
from hiver_support.data.reply_classify import is_probably_english  # noqa: E402
from hiver_support.data.split import temporal_split  # noqa: E402
from discover_taxonomy import BRAND, load_brand_pairs  # noqa: E402

REPORTS = ROOT / "reports"
RANDOM_SEED = 20260910
EXAMPLES_PER_GROUP = 6

# --- Study 1: account access vs security ------------------------------------------------
ACCOUNT_GROUPS: dict[str, str] = {
    "access_credentials": r"\b(forgot(ten)? my (password|passcode|apple ?id)|"
    r"can'?t (log ?in|login|sign ?in|remember my password)|"
    r"reset my password|locked out|account (locked|disabled)|"
    r"two.?factor|2fa|verification code|recovery key|security question)\b",
    "compromise_takeover": r"\b(hacked|hacker|someone (else )?(has|is using|accessed|got into)|"
    r"unauthoriz\w*|unauthoris\w*|compromised|stolen (my )?(account|apple ?id)|"
    r"took over my account|logged in from)\b",
    "phishing_impersonation": r"\b(phish\w*|is this (email|text|message) (from you|legit|real)|"
    r"pretending to be|impersonat\w*|fake (email|apple|message)|"
    r"scam\w*|not from you right)\b",
    "privacy_data": r"\b(privacy|my data|personal (data|information)|data breach|"
    r"track(ing)? me|spy(ing)?|listening to me|selling my data)\b",
}

# --- Study 2: intent vs topic -------------------------------------------------------------
_UPDATE = r"\b(ios ?\d|since (the )?update|after (the )?update|new (ios|update|version)|updated to|upgrade[d]?)\b"
_BATTERY = r"\b(battery|charging|charger|drain\w*|overheat\w*)\b"
_MALFUNCTION = r"\b(not working|doesn'?t work|broken|freez\w*|crash\w*|stuck|glitch\w*|unresponsive)\b"
_CONNECTIVITY = r"\b(wi-?fi|bluetooth|cellular|lte|no service|won'?t connect|hotspot|airdrop)\b"
_APPS = r"\b(apple music|itunes|icloud|imessage|facetime|app ?store|siri|safari|apple ?pay)\b"
_ACCOUNT = r"\b(apple ?id|password|passcode|locked out|2fa|two.?factor|sign ?in|log ?in)\b"
_REPAIR = r"\b(repair\w*|replacement|warranty|applecare|genius bar|cracked|water damage|my order|deliver\w*)\b"
_HOWTO = r"\b(how (do|can|would) i|how to|is there a way|where (do|can) i find)\b"

PAIRS: list[tuple[str, str, str, str, str]] = [
    ("battery_vs_update", "battery_only", f"(?=.*{_BATTERY})(?!.*{_UPDATE})",
     "battery_with_update", f"(?=.*{_BATTERY})(?=.*{_UPDATE})"),
    ("malfunction_vs_update", "malfunction_only", f"(?=.*{_MALFUNCTION})(?!.*{_UPDATE})",
     "malfunction_with_update", f"(?=.*{_MALFUNCTION})(?=.*{_UPDATE})"),
    ("connectivity_vs_account", "connectivity_only", f"(?=.*{_CONNECTIVITY})(?!.*{_ACCOUNT})",
     "account_only", f"(?=.*{_ACCOUNT})(?!.*{_CONNECTIVITY})"),
    ("apps_vs_malfunction", "named_app", f"(?=.*{_APPS})(?!.*{_REPAIR})",
     "device_wide_malfunction", f"(?=.*{_MALFUNCTION})(?!.*{_APPS})(?!.*{_REPAIR})"),
    ("repair_vs_malfunction", "repair_logistics", f"(?=.*{_REPAIR})",
     "malfunction_no_repair", f"(?=.*{_MALFUNCTION})(?!.*{_REPAIR})"),
    ("howto_vs_troubleshooting", "howto_no_fault", f"(?=.*{_HOWTO})(?!.*{_MALFUNCTION})",
     "fault_reported", f"(?=.*{_MALFUNCTION})(?!.*{_HOWTO})"),
]


def _git_sha() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True
        ).stdout.strip() or "unknown"
    except Exception:  # pragma: no cover
        return "unknown"


def _select(texts: list[str], pattern: str) -> np.ndarray:
    compiled = re.compile(pattern, re.IGNORECASE | re.DOTALL)
    return np.array([bool(compiled.search(t)) for t in texts])


def _examples(indices, texts, replies, pairs, rng, n=EXAMPLES_PER_GROUP) -> list[dict]:
    if len(indices) == 0:
        return []
    chosen = rng.choice(indices, size=min(n, len(indices)), replace=False)
    return [
        {
            "pair_id": pairs[i].pair_id,
            "customer": mask_pii(texts[i]).text[:230],
            "brand_reply": mask_pii(replies[i]).text[:230],
        }
        for i in sorted(int(x) for x in chosen)
    ]


def main() -> None:
    pairs = list(temporal_split(load_brand_pairs(None)).train)
    texts = [normalise_text(p.customer_text, brand=BRAND) for p in pairs]
    replies = [normalise_text(p.support_text, brand=BRAND) for p in pairs]

    english = np.array([is_probably_english(t) for t in texts])
    keep = np.flatnonzero(english)
    pairs = [pairs[i] for i in keep]
    texts = [texts[i] for i in keep]
    replies = [replies[i] for i in keep]
    rng = np.random.default_rng(RANDOM_SEED + 21)

    print(f"train (English only): {len(texts):,}\n")

    # ---------------- Study 1 ----------------
    print("=" * 78)
    print("STUDY 1 - account access vs security/privacy")
    print("=" * 78)
    account_masks = {name: _select(texts, pat) for name, pat in ACCOUNT_GROUPS.items()}
    study1 = {}
    for name, mask in account_masks.items():
        idx = np.flatnonzero(mask)
        profile = handling_profile(name, [replies[i] for i in idx])
        study1[name] = {
            "profile": profile.as_dict(),
            "examples": _examples(idx, texts, replies, pairs, rng),
        }
        print(f"  {name:<24} n={profile.n:>5}  defl={profile.deflection_rate:.3f}  "
              f"act={profile.actionable_rate:.3f}  subst={profile.substantive_rate:.3f}  "
              f"ask={profile.asks_question_rate:.3f}")

    print("\n  pairwise (bootstrap 95% CI; groups <"
          f"{MIN_GROUP_FOR_VERDICT} give no verdict):")
    names = list(ACCOUNT_GROUPS)
    study1_comparisons = {}
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            a_idx, b_idx = np.flatnonzero(account_masks[a]), np.flatnonzero(account_masks[b])
            diffs = compare_handling(
                [replies[i] for i in a_idx], [replies[i] for i in b_idx], seed=RANDOM_SEED
            )
            study1_comparisons[f"{a} vs {b}"] = [asdict(d) for d in diffs]
            flags = [f"{d.metric.split('_')[0]}{'*' if d.significant else ''}" for d in diffs]
            sig = [d for d in diffs if d.significant]
            print(f"    {a:<22} vs {b:<24} significant: "
                  f"{[d.metric for d in sig] if sig else 'NONE'}")

    # ---------------- Study 2 ----------------
    print("\n" + "=" * 78)
    print("STUDY 2 - intent vs topic/symptom")
    print("=" * 78)
    study2 = {}
    for key, a_name, a_pat, b_name, b_pat in PAIRS:
        a_idx = np.flatnonzero(_select(texts, a_pat))
        b_idx = np.flatnonzero(_select(texts, b_pat))
        a_prof = handling_profile(a_name, [replies[i] for i in a_idx])
        b_prof = handling_profile(b_name, [replies[i] for i in b_idx])
        diffs = compare_handling(
            [replies[i] for i in a_idx], [replies[i] for i in b_idx], seed=RANDOM_SEED
        )
        sig = [d for d in diffs if d.significant]
        study2[key] = {
            "a": a_prof.as_dict(),
            "b": b_prof.as_dict(),
            "differences": [asdict(d) for d in diffs],
            "significant_metrics": [d.metric for d in sig],
            "a_examples": _examples(a_idx, texts, replies, pairs, rng),
            "b_examples": _examples(b_idx, texts, replies, pairs, rng),
        }
        print(f"\n  {key}")
        print(f"    {a_name:<26} n={a_prof.n:>6}  defl={a_prof.deflection_rate:.3f} "
              f"act={a_prof.actionable_rate:.3f} subst={a_prof.substantive_rate:.3f}")
        print(f"    {b_name:<26} n={b_prof.n:>6}  defl={b_prof.deflection_rate:.3f} "
              f"act={b_prof.actionable_rate:.3f} subst={b_prof.substantive_rate:.3f}")
        for d in diffs:
            if d.significant:
                print(f"      * {d.metric:<22} diff={d.difference:+.3f} "
                      f"CI[{d.ci_low:+.3f}, {d.ci_high:+.3f}]")
        if not sig:
            print("      (no significant difference on any metric)")

    artifact = {
        "provenance": {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "script": "scripts/adjudicate_taxonomy.py",
            "git_sha": _git_sha(),
            "brand": BRAND,
            "split": "train (English only)",
            "n": len(texts),
            "random_seed": RANDOM_SEED,
            "min_group_for_verdict": MIN_GROUP_FOR_VERDICT,
            "note": (
                "Support replies are used to infer OPERATIONAL HANDLING for taxonomy design. "
                "They are never used to assign labels during annotation. Train split only."
            ),
        },
        "study_1_account_vs_security": {
            "groups": study1,
            "comparisons": study1_comparisons,
        },
        "study_2_intent_vs_topic": study2,
    }
    REPORTS.mkdir(exist_ok=True)
    out = REPORTS / "taxonomy_adjudication.json"
    out.write_text(json.dumps(artifact, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
