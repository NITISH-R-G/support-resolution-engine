"""Convert evaluation and golden artifacts between full-text and text-free form.

Redaction replaces every tweet-derived string with its sha256 **in the same key position**, and
restoration puts verified text back, so ``restore(redact(row), texts)`` serialises to exactly the
original bytes. That is what lets a materialised file be checked against the hash of the
original artifact, not merely field by field.

Text-derived features that analyses need are computed once at redaction time and stored beside
the hashes (``derived_text_features``); restoration removes them.
"""
from __future__ import annotations

import hashlib
import re

_DERIVED = "derived_text_features"


def sha256_text(text: str | None) -> str | None:
    return None if text is None else hashlib.sha256(text.encode("utf-8")).hexdigest()


class TextMismatchError(ValueError):
    """A reconstructed text does not match the committed hash."""


def _check(label: str, text: str | None, digest: str | None) -> str | None:
    if sha256_text(text) != digest:
        raise TextMismatchError(f"{label}: reconstructed text does not match the committed sha256")
    return text


# ----------------------------------------------------------------------------------- predictions

def prediction_features(row: dict, deflection_tail_re: re.Pattern[str]) -> dict:
    """The only reply/evidence properties failure analysis uses, frozen at redaction time."""
    reply = row.get("reply") or ""
    evidence = row.get("evidence") or []
    return {
        "reply_dm_style": "dm" in reply.lower().split() or " dm" in reply.lower(),
        "reply_deflection_tail": bool(deflection_tail_re.search(reply)) if reply else False,
        "evidence_deflection_tail": any(deflection_tail_re.search(e["resolution_text"]) for e in evidence),
        "evidence_mentions_dm": any(" dm" in e["resolution_text"].lower() for e in evidence),
    }


def redact_prediction(row: dict, deflection_tail_re: re.Pattern[str]) -> dict:
    out = {}
    for key, value in row.items():
        if key == "message":
            out["message_sha256"] = sha256_text(value)
        elif key == "reply":
            out["reply_sha256"] = sha256_text(value)
        elif key == "evidence":
            out["evidence"] = [
                {
                    (k + "_sha256" if k in ("customer_text", "resolution_text") else k):
                    (sha256_text(v) if k in ("customer_text", "resolution_text") else v)
                    for k, v in item.items()
                }
                for item in value
            ]
        else:
            out[key] = value
    out[_DERIVED] = prediction_features(row, deflection_tail_re)
    return out


def restore_prediction(row: dict, message: str, reply: str | None, evidence_texts: list[tuple[str, str]]) -> dict:
    """Inverse of ``redact_prediction``; raises ``TextMismatchError`` on any hash mismatch."""
    label = f"{row.get('system')}:{row.get('pair_id')}"
    if len(evidence_texts) != len(row.get("evidence", [])):
        raise TextMismatchError(f"{label}: evidence count differs")
    out = {}
    for key, value in row.items():
        if key == _DERIVED:
            continue
        if key == "message_sha256":
            out["message"] = _check(f"{label} message", message, value)
        elif key == "reply_sha256":
            out["reply"] = _check(f"{label} reply", reply, value)
        elif key == "evidence":
            restored = []
            for item, (customer_text, resolution_text) in zip(value, evidence_texts):
                new = {}
                for k, v in item.items():
                    if k == "customer_text_sha256":
                        new["customer_text"] = _check(f"{label} evidence", customer_text, v)
                    elif k == "resolution_text_sha256":
                        new["resolution_text"] = _check(f"{label} evidence", resolution_text, v)
                    else:
                        new[k] = v
                restored.append(new)
            out["evidence"] = restored
        else:
            out[key] = value
    return out


# ---------------------------------------------------------------------------------------- judge

def redact_judgement(row: dict) -> dict:
    return {("rationale_sha256" if k == "rationale" else k): (sha256_text(v) if k == "rationale" else v)
            for k, v in row.items()}


def restore_judgement(row: dict, rationale: str) -> dict:
    label = f"judge {row.get('system')}:{row.get('pair_id')}"
    return {("rationale" if k == "rationale_sha256" else k):
            (_check(label, rationale, v) if k == "rationale_sha256" else v)
            for k, v in row.items()}


# ----------------------------------------------------------------------------------- candidates

def candidate_record(v1: dict, v2: dict, context_tweet_ids: list[str]) -> dict:
    """Text-free committed record carrying the text hashes of both golden-set versions."""
    if {k: v for k, v in v1.items() if k not in ("customer_message", "context")} != \
            {k: v for k, v in v2.items() if k not in ("customer_message", "context")}:
        raise ValueError(f"{v1['pair_id']}: v1 and v2 differ outside the text fields")
    if len(v1["context"]) != len(context_tweet_ids):
        raise ValueError(f"{v1['pair_id']}: context turn count does not match its tweet ids")
    out = {}
    for key, value in v1.items():
        if key == "customer_message":
            continue
        if key == "context":
            out["context"] = [
                {k: val for k, val in turn.items() if k != "text"} | {"tweet_id": tweet_id}
                for turn, tweet_id in zip(value, context_tweet_ids)
            ]
        else:
            out[key] = value
    out["text_sha256"] = {
        version: {
            "customer_message": sha256_text(record["customer_message"]),
            "context": [sha256_text(turn["text"]) for turn in record["context"]],
        }
        for version, record in (("v1", v1), ("v2", v2))
    }
    return out


def verify_candidate(record: dict, rebuilt: dict, version: str) -> None:
    """Check a rebuilt full-text candidate against the committed record. Raises on mismatch."""
    hashes = record["text_sha256"][version]
    label = f"candidate {record['pair_id']} ({version})"
    _check(f"{label} message", rebuilt["customer_message"], hashes["customer_message"])
    if len(rebuilt["context"]) != len(hashes["context"]):
        raise TextMismatchError(f"{label}: context turn count differs")
    for turn, digest in zip(rebuilt["context"], hashes["context"]):
        _check(f"{label} context", turn["text"], digest)
    stripped = {k: v for k, v in record.items() if k not in ("text_sha256", "context")}
    for key, value in stripped.items():
        if rebuilt.get(key) != value:
            raise TextMismatchError(f"{label}: field {key!r} differs")
