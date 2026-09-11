"""Blind annotation tool for the golden set. The only way a gold label enters this project.

**This script is structurally blind.** It imports no classifier, no retriever and no
generator, and the candidate records it reads have no field for the brand's reply or for a
label. So there is nothing here to anchor on: not a suggestion, not a confidence, not a
retrieved case, not the answer. That is the protocol's pass-1 requirement
(``docs/GOLDEN_SET.md`` section 7) enforced by what the file imports rather than by memory.

``tests/test_golden_annotation.py`` asserts the import list, so adding "just a hint" to speed
annotation up fails the build.

It cannot be used to bulk-fill. Every label is typed, one example at a time, and the log is
append-only so a later pass never erases an earlier one — disagreement between passes is the
measurement.

Usage:
    python scripts/annotate_golden.py --annotator nitish
    python scripts/annotate_golden.py --annotator nitish --pass 2 --limit 40
    python scripts/annotate_golden.py --status
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hiver_support.golden.schema import (  # noqa: E402
    ExpectedResolutionKind,
    GoldenAnnotation,
    GoldenSetError,
    LabelConfidence,
)
from hiver_support.golden.store import (  # noqa: E402
    append_annotation,
    coverage,
    latest_by_pair,
    merge,
    read_annotations,
    read_candidates,
)
from hiver_support.taxonomy import TAXONOMY  # noqa: E402

GOLDEN_DIR = ROOT / "data" / "golden"
CANDIDATES = GOLDEN_DIR / "candidates.jsonl"
ANNOTATIONS = GOLDEN_DIR / "annotations.jsonl"

RESOLUTION_KINDS = tuple(ExpectedResolutionKind)
RULE = "=" * 78


def _safe(text: str) -> str:
    """Windows consoles are cp1252; an emoji must not crash an annotation session."""
    return text.encode(sys.stdout.encoding or "utf-8", "replace").decode(
        sys.stdout.encoding or "utf-8", "replace"
    )


def _ask(prompt: str) -> str:
    try:
        return input(prompt).strip()
    except EOFError:
        # Piped stdin means someone is trying to bulk-fill. Gold labels are typed.
        raise GoldenSetError(
            "stdin closed: annotation is interactive by design and cannot be piped"
        ) from None


def _ask_flag(prompt: str, default: bool) -> bool:
    hint = "Y/n" if default else "y/N"
    while True:
        answer = _ask(f"  {prompt} [{hint}]: ").lower()
        if not answer:
            return default
        if answer in ("y", "yes"):
            return True
        if answer in ("n", "no"):
            return False
        print("    answer y or n")


def _ask_choice(prompt: str, options: list[str], default: str | None = None) -> str:
    while True:
        answer = _ask(f"  {prompt}: ").strip()
        if not answer and default is not None:
            return default
        if answer.isdigit() and 1 <= int(answer) <= len(options):
            return options[int(answer) - 1]
        if answer in options:
            return answer
        print(f"    choose 1-{len(options)} or type the name exactly")


def _show_intents() -> None:
    print("\n  INTENTS")
    for index, intent in enumerate(TAXONOMY.intents, start=1):
        marker = "!" if intent.escalation_sensitive else " "
        print(f"   {index:>2}{marker} {intent.name:<26} {intent.definition[:46]}")
    print("   (! = escalation-sensitive by policy; that is the POLICY's view, not yours)")
    print("   ('?' for the full codebook - definitions, inclusions, exclusions)")


def _show_codebook() -> None:
    """The full frozen definitions, on demand.

    The one-line list above is truncated to keep the screen readable, and a truncated
    codebook is how an annotator guesses. Over 200 examples a misread definition is the most
    expensive error available: it produces confidently wrong gold, and every number computed
    against it inherits the mistake.
    """
    print("\n" + RULE)
    print(f"CODEBOOK - frozen taxonomy {TAXONOMY.version} ({TAXONOMY.frozen_hash[:16]}...)")
    print(RULE)
    for index, intent in enumerate(TAXONOMY.intents, start=1):
        flag = "  [escalation-sensitive by policy]" if intent.escalation_sensitive else ""
        print(f"\n  {index}. {intent.name}{flag}")
        print(f"     {_safe(intent.definition)}")
        if intent.includes:
            print(f"     INCLUDES: {_safe('; '.join(intent.includes))}")
        if intent.excludes:
            print(f"     EXCLUDES: {_safe('; '.join(intent.excludes))}")
        if intent.confusions:
            print(f"     OFTEN CONFUSED WITH: {', '.join(intent.confusions)}")
    print("\n  ATTRIBUTES (independent of the intent - a message can be any intent AND these)")
    for attribute in TAXONOMY.attributes:
        print(f"\n  {attribute.name}")
        print(f"     {_safe(attribute.definition)}")
        print(f"     RULE: {_safe(attribute.annotation_rule)}")
    print("\n  Full guide: docs/ANNOTATION_GUIDE.md")
    print(RULE)


def _show_candidate(candidate, position: int, total: int) -> None:
    print("\n" + RULE)
    print(f"[{position}/{total}]  {candidate.pair_id}   {candidate.created_at:%Y-%m-%d %H:%M}")
    print(RULE)
    if candidate.context:
        print(f"\n  EARLIER IN THIS CONVERSATION ({len(candidate.context)} turn(s))")
        for turn in candidate.context:
            print(f"    {turn.author_role:>8}: {_safe(turn.text)[:300]}")
    print("\n  CUSTOMER MESSAGE")
    for line in _safe(candidate.customer_message).splitlines() or [""]:
        print(f"    {line}")


def _annotate_one(candidate, annotator: str, pass_number: int) -> GoldenAnnotation | None:
    started = time.time()
    _show_intents()
    names = list(TAXONOMY.names)
    while True:
        intent = _ask_choice("intent (number, name, '?' codebook, 's' skip)", names + ["s", "?"])
        if intent != "?":
            break
        _show_codebook()
        # Time spent reading the codebook is not time spent deciding, so it does not count
        # towards this example's duration.
        started = time.time()
    if intent == "s":
        return None

    security = _ask_flag("security_sensitive?", default=False)
    context_ok = _ask_flag("context_sufficient?", default=True)
    escalate = _ask_flag("should_escalate? (your judgement, not the policy's)", default=False)

    print("\n  EXPECTED RESOLUTION KIND")
    for index, kind in enumerate(RESOLUTION_KINDS, start=1):
        print(f"   {index:>2}  {kind.value}")
    kind = _ask_choice("resolution kind", [k.value for k in RESOLUTION_KINDS])

    ambiguous = _ask_flag("is this genuinely ambiguous?", default=False)
    alternative = None
    if ambiguous:
        answer = _ask_choice("runner-up intent (or 'none')", names + ["none"], default="none")
        alternative = None if answer == "none" else answer

    confidence = _ask_choice(
        "confidence [1 high, 2 medium, 3 low]",
        [c.value for c in LabelConfidence],
        default=LabelConfidence.HIGH.value,
    )
    reference = _ask("  what must a good reply contain? (optional): ")
    notes = _ask("  notes (optional): ")

    return GoldenAnnotation(
        pair_id=candidate.pair_id,
        annotator_id=annotator,
        intent=intent,
        security_sensitive=security,
        context_sufficient=context_ok,
        should_escalate=escalate,
        expected_resolution_kind=ExpectedResolutionKind(kind),
        is_ambiguous=ambiguous,
        alternative_intent=alternative,
        label_confidence=LabelConfidence(confidence),
        reference_resolution=reference,
        notes=notes,
        pass_number=pass_number,
        seconds_spent=round(time.time() - started, 1),
    )


def _print_status() -> None:
    candidates = read_candidates(CANDIDATES)
    annotations = read_annotations(ANNOTATIONS)
    stats = coverage(merge(candidates, annotations))
    print(f"Golden set: {CANDIDATES}")
    print(f"  total candidates   {stats['total']}")
    print(f"  HUMAN_LABELED      {stats['human_labelled']}")
    print(f"  UNLABELED          {stats['unlabelled']}")
    print(f"  WEAKLY_LABELED     {stats['weakly_labelled']}  (never permitted as gold)")
    print(f"  MODEL_GENERATED    {stats['model_generated']}  (never permitted as gold)")
    for pass_number in sorted({a.pass_number for a in annotations}):
        done = len(latest_by_pair(annotations, pass_number=pass_number))
        print(f"  pass {pass_number}: {done} annotated")
    if not stats["complete"]:
        print(
            f"\n  The set is NOT complete. load_gold() will refuse to score it, and no label "
            f"will be backfilled from weak rules, the classifier or an LLM."
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotator", help="your name; gold labels are attributed")
    parser.add_argument("--pass", dest="pass_number", type=int, default=1)
    parser.add_argument("--limit", type=int, default=None, help="stop after N examples")
    parser.add_argument("--status", action="store_true", help="show progress and exit")
    args = parser.parse_args()

    if args.status:
        _print_status()
        return
    if not args.annotator:
        parser.error("--annotator is required: gold labels must be attributed to a human")

    candidates = read_candidates(CANDIDATES)
    existing = latest_by_pair(read_annotations(ANNOTATIONS), pass_number=args.pass_number)
    todo = [c for c in candidates if c.pair_id not in existing]
    if args.limit:
        todo = todo[: args.limit]

    print(RULE)
    print(f"BLIND ANNOTATION - pass {args.pass_number} - annotator {args.annotator!r}")
    print(RULE)
    print(f"  taxonomy {TAXONOMY.version} ({TAXONOMY.frozen_hash[:16]}...)")
    print(f"  {len(existing)} already done this pass, {len(todo)} to go")
    print("  You will NOT see: the brand's reply, any model prediction, or any suggestion.")
    print("  Guide: docs/ANNOTATION_GUIDE.md   Protocol: docs/GOLDEN_SET.md")
    print("  Enter 's' at the intent prompt to skip, Ctrl-C to stop. Progress is saved as")
    print("  you go, so stopping loses nothing.")

    done = 0
    for position, candidate in enumerate(todo, start=1):
        _show_candidate(candidate, position, len(todo))
        try:
            annotation = _annotate_one(candidate, args.annotator, args.pass_number)
        except KeyboardInterrupt:
            print("\n\nStopped. Everything answered so far is saved.")
            break
        if annotation is None:
            print("  skipped.")
            continue
        append_annotation(ANNOTATIONS, annotation)
        done += 1
        print(f"  saved ({annotation.seconds_spent:.0f}s).")

    print(f"\n{done} annotation(s) written to {ANNOTATIONS}")
    _print_status()


if __name__ == "__main__":
    main()
