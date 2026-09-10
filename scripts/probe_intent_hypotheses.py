"""Measure prevalence of hypothesised intent categories in the AppleSupport train split.

**Why this exists.** Embedding clusters over this corpus are dominated by a single temporal
event (the iOS 11 launch): at k=12 and k=16 the majority of clusters are variants of "the
update broke my phone". Operationally important but lower-frequency categories — billing,
account access, orders, repair, security — do not surface as clusters at all. Clustering
alone therefore cannot test the category hypotheses in `SPEC.md` §5, so this script measures
each hypothesis directly.

**These probes are MEASUREMENT, not labelling.** A probe pattern must never become a
classifier or a gold label. One audited public repository generated its golden labels by
keyword matching and then reported that its keyword baseline "beat" a trained model — the
baseline *was* the labelling function (`docs/PUBLIC_REPO_COMPARISON.md` §2.3). Probe output
here is evidence for a human deciding which categories exist, with what support, and how much
they overlap.

Probes are deliberately **high-precision, low-recall**: they undercount. A probe count is a
floor on prevalence, never an estimate of it, and is reported as such.

Usage:
    python scripts/probe_intent_hypotheses.py
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hiver_support.data.normalise import normalise_text  # noqa: E402
from hiver_support.data.pii import mask_pii  # noqa: E402
from hiver_support.data.reply_classify import classify_reply, is_probably_english  # noqa: E402
from hiver_support.data.split import temporal_split  # noqa: E402

sys.path.insert(0, str(ROOT / "scripts"))
from discover_taxonomy import BRAND, load_brand_pairs  # noqa: E402

REPORTS = ROOT / "reports"
RANDOM_SEED = 20260910
EXAMPLES_PER_PROBE = 10

# --- hypothesis probes --------------------------------------------------------------------
# Each entry tests ONE hypothesised category from SPEC.md §5 / the milestone brief.
# Patterns are conservative: they aim to be right when they fire, not to catch everything.

PROBES: dict[str, str] = {
    # --- billing / payment ---
    "billing_payment": r"\b(charged|charge|billed|billing|invoice|receipt|refund|money back|"
    r"overcharg|double.?charg|payment (method|failed|declined)|card (declined|charged)|"
    r"unauthorized charge|unauthorised charge)\b",
    # --- subscription lifecycle (distinct from a one-off charge) ---
    "subscription": r"\b(subscription|subscribe[dr]?|unsubscribe|cancel(ling|led)? my (plan|"
    r"membership|subscription)|auto.?renew|free trial|trial (ended|period)|"
    r"icloud storage plan|apple music (plan|subscription))\b",
    # --- account access ---
    "account_access": r"\b(apple ?id|password|passcode|locked out|can'?t (log ?in|sign ?in|"
    r"access my account)|forgot my (password|apple id)|two.?factor|2fa|verification code|"
    r"security question|recovery key|account (disabled|locked))\b",
    # --- security / privacy / fraud ---
    "security_privacy": r"\b(hacked|hacker|phish|phishing|scam|scammed|fraud|"
    r"unauthoriz|unauthoris|someone (else )?(has|is using|accessed)|stolen (id|account)|"
    r"data breach|privacy|spy(ing)?|track(ing)? me)\b",
    # --- orders / delivery ---
    "order_delivery": r"\b(my order|order (number|status|placed|cancelled|canceled)|"
    r"deliver(y|ed|ing)|shipping|shipped|dispatch|tracking (number|info)|"
    r"still (hasn'?t|has not) arrived|when will (it|my order) arrive)\b",
    # --- repair / replacement / warranty ---
    "repair_replacement": r"\b(repair|replacement|replaced|warranty|applecare|genius bar|"
    r"service centre|service center|sent (it )?(in|off) for repair|"
    r"screen (repair|replacement)|out of warranty|rma)\b",
    # --- physical hardware damage / defect ---
    "hardware_physical": r"\b(cracked|shattered|broken screen|water damage|liquid damage|"
    r"bent|swollen battery|won'?t (turn on|charge)|dead (phone|device)|"
    r"port (broken|not working)|button (stuck|broken)|speaker (broken|not working))\b",
    # --- connectivity ---
    "connectivity": r"\b(wi-?fi|bluetooth|cellular|lte|4g|5g|no service|"
    r"won'?t connect|can'?t connect|airdrop|hotspot|network (issue|problem))\b",
    # --- battery / charging ---
    "battery_charging": r"\b(battery|charging|charger|drain(ing|s|ed)?|"
    r"percent(age)? (drop|drain)|dies (fast|quickly)|won'?t hold (a )?charge)\b",
    # --- software update / OS behaviour ---
    "software_update": r"\b(ios ?\d|update[ds]?|updating|upgrade[d]?|"
    r"new (version|software)|since the update|after (the )?update|beta|downgrade)\b",
    # --- app / service behaviour ---
    "apps_services": r"\b(apple music|itunes|icloud|imessage|facetime|app ?store|"
    r"siri|safari|photos app|mail app|podcast|apple ?tv|apple ?pay|app (crash|keeps closing))\b",
    # --- explicit how-to / information request ---
    "howto_information": r"\b(how (do|can|would) i|how to|is there a way to|"
    r"what does .{0,20}mean|can i |do i need to|where (do|can) i find|"
    r"what'?s the difference|does (it|the iphone) support)\b",
    # --- complaint / feedback with no actionable request ---
    "complaint_feedback": r"\b(worst|disgrace|disgusting|ridiculous|unacceptable|"
    r"never buying|switching to android|shame on you|garbage|rubbish|"
    r"terrible (service|support)|useless|joke)\b",
}

# Signals that a message cannot be classified from the customer text alone.
#
# ANCHORING NOTE — this cost a measurement. `normalise_text` deliberately PRESERVES the brand
# handle, because it identifies the addressee, so a normalised message reads
# "@AppleSupport Ok" rather than "Ok". The first version of these patterns anchored on ^ with
# only a [USER] prefix allowed and reported **zero** bare acknowledgements — while the k=12
# clustering showed ~8% of the split is exactly that ("@AppleSupport Ok", "@AppleSupport 11",
# "@AppleSupport Thanks"). The leading group below is what makes these probes measure the
# phenomenon instead of the punctuation.
_LEAD = r"^[\s\W]*(?:(?:@\w+|\[USER\])[\s,:.]*)*"

INSUFFICIENT_CONTEXT_PROBES: dict[str, str] = {
    "url_only_or_screenshot": _LEAD + r"(?:\[URL\][\s!?.]*)+$|"
    + _LEAD + r"(this|it|here|look|wtf|help|fix (this|it))[\s!?.]*\[URL\][\s!?.]*$",
    "bare_acknowledgement": _LEAD + r"(ok(ay)?|yes+|no|yep|nope|thanks?( you)?|thank you|"
    r"sure|done|k|\d{1,3})[\s!.?]*$",
    "continuation_reply": _LEAD + r"(i (did|have|already|tried|updated)|it (still|didn'?t|does)|"
    r"still (not|doesn'?t|nothing)|that (didn'?t|doesn'?t)|(yes|no),? i|already (did|tried))\b",
}


def _git_sha() -> str:
    try:
        return (
            subprocess.run(
                ["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True
            ).stdout.strip()
            or "unknown"
        )
    except Exception:  # pragma: no cover
        return "unknown"


def main() -> None:
    pairs = load_brand_pairs(None)
    train = list(temporal_split(pairs).train)
    texts = [normalise_text(p.customer_text, brand=BRAND) for p in train]
    total = len(texts)
    rng = np.random.default_rng(RANDOM_SEED + 11)

    compiled = {name: re.compile(pattern, re.IGNORECASE) for name, pattern in PROBES.items()}
    compiled_ctx = {
        name: re.compile(pattern, re.IGNORECASE)
        for name, pattern in INSUFFICIENT_CONTEXT_PROBES.items()
    }

    hits: dict[str, list[int]] = {name: [] for name in compiled}
    for index, text in enumerate(texts):
        for name, pattern in compiled.items():
            if pattern.search(text):
                hits[name].append(index)

    ctx_hits = {
        name: [i for i, t in enumerate(texts) if pattern.search(t)]
        for name, pattern in compiled_ctx.items()
    }
    non_english = [i for i, t in enumerate(texts) if not is_probably_english(t)]

    def summarise(name: str, indices: list[int]) -> dict:
        replies = [
            classify_reply(normalise_text(train[i].support_text, brand=BRAND)) for i in indices
        ] or []
        chosen = (
            rng.choice(indices, size=min(EXAMPLES_PER_PROBE, len(indices)), replace=False)
            if indices
            else []
        )
        return {
            "probe": name,
            "count": len(indices),
            "share_of_train": round(len(indices) / total, 4),
            "reply_deflection_rate": round(float(np.mean([r.is_deflection for r in replies])), 3)
            if replies
            else None,
            "reply_actionable_rate": round(float(np.mean([r.is_actionable for r in replies])), 3)
            if replies
            else None,
            "mean_words": round(float(np.mean([len(texts[i].split()) for i in indices])), 1)
            if indices
            else None,
            "examples": [
                {
                    "pair_id": train[i].pair_id,
                    "customer": mask_pii(texts[i]).text[:240],
                    "brand_reply": mask_pii(
                        normalise_text(train[i].support_text, brand=BRAND)
                    ).text[:240],
                }
                for i in sorted(int(x) for x in chosen)
            ],
        }

    probe_results = [summarise(name, indices) for name, indices in hits.items()]
    probe_results.sort(key=lambda r: -r["count"])

    # Overlap between probes: how often does one message match several categories? This is the
    # multi-intent and label-confusability signal that decides merges and tie-break rules.
    membership = {name: set(indices) for name, indices in hits.items()}
    per_message_counts = Counter(
        sum(1 for s in membership.values() if index in s) for index in range(total)
    )

    pair_overlap = {}
    names = list(membership)
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            inter = len(membership[a] & membership[b])
            if inter >= 200:
                union = len(membership[a] | membership[b])
                pair_overlap[f"{a} & {b}"] = {
                    "both": inter,
                    "jaccard": round(inter / union, 3),
                    "share_of_a": round(inter / max(len(membership[a]), 1), 3),
                    "share_of_b": round(inter / max(len(membership[b]), 1), 3),
                }
    top_overlap = dict(
        sorted(pair_overlap.items(), key=lambda kv: -kv[1]["jaccard"])[:20]
    )

    artifact = {
        "provenance": {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "script": "scripts/probe_intent_hypotheses.py",
            "git_sha": _git_sha(),
            "brand": BRAND,
            "split": "train",
            "train_size": total,
            "random_seed": RANDOM_SEED,
            "note": (
                "High-precision, low-recall lexical probes. Counts are FLOORS on prevalence, "
                "not estimates. These measure hypotheses; they never assign labels."
            ),
        },
        "probes": probe_results,
        "insufficient_context": [summarise(n, i) for n, i in ctx_hits.items()],
        "non_english": {
            "count": len(non_english),
            "share_of_train": round(len(non_english) / total, 4),
        },
        "messages_matching_n_probes": {
            str(k): v for k, v in sorted(per_message_counts.items())
        },
        "probe_overlap_top20_by_jaccard": top_overlap,
    }

    REPORTS.mkdir(exist_ok=True)
    out = REPORTS / "taxonomy_probes.json"
    out.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")

    print(f"train messages: {total:,}\n")
    print(f"{'probe':<24} {'count':>7} {'share':>8} {'defl':>6} {'act':>6} {'words':>6}")
    for r in probe_results:
        print(
            f"{r['probe']:<24} {r['count']:>7,} {r['share_of_train']:>7.1%} "
            f"{r['reply_deflection_rate'] or 0:>6.2f} {r['reply_actionable_rate'] or 0:>6.2f} "
            f"{r['mean_words'] or 0:>6.1f}"
        )
    print("\ninsufficient-context probes:")
    for r in artifact["insufficient_context"]:
        print(f"  {r['probe']:<24} {r['count']:>7,} {r['share_of_train']:>7.1%}")
    print(f"  {'non_english':<24} {len(non_english):>7,} "
          f"{artifact['non_english']['share_of_train']:>7.1%}")
    print("\nmessages matching N probes:", artifact["messages_matching_n_probes"])
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
