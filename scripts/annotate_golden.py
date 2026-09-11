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
    FlagRecord,
    LabelConfidence,
    Retraction,
    ReviewAction,
)
from hiver_support.golden.store import (  # noqa: E402
    append_annotation,
    append_flag,
    append_retraction,
    load_effective_annotations,
    coverage,
    latest_by_pair,
    merge,
    read_annotations,
    read_candidates,
    read_retractions,
    unresolved_pair_ids,
)
from hiver_support.golden.suggestions import (  # noqa: E402
    ModelSuggestion,
    blind_pair_ids,
    needs_mandatory_review,
    read_suggestions,
)
from hiver_support.taxonomy import TAXONOMY  # noqa: E402

GOLDEN_DIR = ROOT / "data" / "golden"
CANDIDATES = GOLDEN_DIR / "candidates.jsonl"
ANNOTATIONS = GOLDEN_DIR / "annotations.jsonl"
SUGGESTIONS = GOLDEN_DIR / "suggestions.jsonl"

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


# Compact edits for the forced-review cases, which were costing six keystrokes each - `C`,
# then Enter through five fields - even when a single field was wrong.
_FLAG_TOKENS = {
    "e": "should_escalate",
    "sec": "security_sensitive",
    "ctx": "context_sufficient",
}


def _parse_inline_edit(command: str, suggestion, intent_names: list[str]):
    """Parse a one-line correction such as ``i5`` or ``i5 e``.

    Returns ``(values, changed_fields)``, or ``None`` when the input is one of the existing
    A/C/F/S actions rather than an edit.

    Raises:
        ValueError: when the input looks like an edit but is not valid. Guessing at a
            malformed correction is how a mistyped keystroke becomes a gold label.
    """
    text = (command or "").strip().lower()
    if not text or text in ("a", "c", "f", "s", "?", "accept", "correct", "flag", "skip"):
        return None

    values = {
        "intent": suggestion.intent,
        "security_sensitive": suggestion.security_sensitive,
        "context_sufficient": suggestion.context_sufficient,
        "should_escalate": suggestion.should_escalate,
        "expected_resolution_kind": suggestion.expected_resolution_kind,
    }
    changed: list[str] = []

    for token in text.split():
        if token in _FLAG_TOKENS:
            field = _FLAG_TOKENS[token]
            values[field] = not values[field]
            if values[field] != getattr(suggestion, field):
                changed.append(field)
            else:
                changed = [c for c in changed if c != field]
            continue

        if token.startswith("i"):
            raw = token[2:] if token[1:2] == "=" else token[1:]
            if not raw:
                raise ValueError(f"{token!r}: give an intent number or name, e.g. i5 or i=connectivity")
            if raw.isdigit():
                index = int(raw)
                if not 1 <= index <= len(intent_names):
                    raise ValueError(f"{token!r}: intent number must be 1-{len(intent_names)}")
                chosen = intent_names[index - 1]
            elif raw in intent_names:
                chosen = raw
            else:
                raise ValueError(f"{token!r}: {raw!r} is not an intent name")
            values["intent"] = chosen
            if chosen != suggestion.intent:
                changed.append("intent")
            else:
                changed = [c for c in changed if c != "intent"]
            continue

        raise ValueError(f"{token!r} is not a recognised edit (use i<N>, i=<name>, e, sec, ctx)")

    return values, tuple(changed)


def _filter_group(candidates, group: str, blind: set[str]):
    """Filter the annotation queue only. The split itself is fixed by ``blind_pair_ids``."""
    if group == "assisted":
        return [c for c in candidates if c.pair_id not in blind]
    if group == "blind":
        return [c for c in candidates if c.pair_id in blind]
    return list(candidates)


def _progress_line(done: int, total: int, forced_left: int) -> str:
    remaining = max(total - done, 0)
    return f"  [{done}/{total}] {remaining} left, {forced_left} needing full review"


def _show_suggestion(suggestion: ModelSuggestion, forced: bool, reasons: tuple) -> None:
    """Display a provisional suggestion, unmistakably as a suggestion."""
    print("\n  " + "-" * 74)
    print("  MODEL SUGGESTION - PROVISIONAL, NOT A LABEL. Nothing is recorded until you act.")
    print(f"  (pre-annotator: {suggestion.provider}:{suggestion.model})")
    print("  " + "-" * 74)
    print(f"    intent                   {suggestion.intent}")
    print(f"    security_sensitive       {suggestion.security_sensitive}")
    print(f"    context_sufficient       {suggestion.context_sufficient}")
    print(f"    should_escalate          {suggestion.should_escalate}")
    print(f"    expected_resolution_kind {suggestion.expected_resolution_kind}")
    print(f"    model confidence         {suggestion.confidence:.2f}")
    if suggestion.rationale:
        print(f"    rationale                {_safe(suggestion.rationale)}")
    if forced:
        print("\n  ** ONE-KEY ACCEPT DISABLED - this example needs your judgement **")
        for reason in reasons:
            print(f"     - {reason}")


def _correct(suggestion: ModelSuggestion, names: list[str]) -> tuple[dict, tuple[str, ...]]:
    """Change only the fields that are wrong. Blank keeps the suggested value."""
    values = {
        "intent": suggestion.intent,
        "security_sensitive": suggestion.security_sensitive,
        "context_sufficient": suggestion.context_sufficient,
        "should_escalate": suggestion.should_escalate,
        "expected_resolution_kind": suggestion.expected_resolution_kind,
    }
    changed: list[str] = []
    print("\n  Correct only what is wrong. Press Enter to keep the suggested value.")

    answer = _ask(f"  intent [{values['intent']}]: ").strip()
    if answer:
        if answer.isdigit() and 1 <= int(answer) <= len(names):
            answer = names[int(answer) - 1]
        if answer not in names:
            print(f"    {answer!r} is not a valid intent; keeping {values['intent']}")
        elif answer != values["intent"]:
            values["intent"] = answer
            changed.append("intent")

    for flag in ("security_sensitive", "context_sufficient", "should_escalate"):
        answer = _ask(f"  {flag} [{values[flag]}] (y/n, Enter keeps): ").strip().lower()
        if answer in ("y", "yes", "n", "no"):
            new = answer in ("y", "yes")
            if new != values[flag]:
                values[flag] = new
                changed.append(flag)

    kinds = [k.value for k in RESOLUTION_KINDS]
    answer = _ask(f"  resolution kind [{values['expected_resolution_kind']}]: ").strip()
    if answer:
        if answer.isdigit() and 1 <= int(answer) <= len(kinds):
            answer = kinds[int(answer) - 1]
        if answer in kinds and answer != values["expected_resolution_kind"]:
            values["expected_resolution_kind"] = answer
            changed.append("expected_resolution_kind")

    return values, tuple(changed)


def _review_one(candidate, suggestion, annotator: str, pass_number: int):
    """Assisted review of one candidate. Returns an annotation, a FlagRecord, or None to skip."""
    started = time.time()
    forced, reasons = needs_mandatory_review(suggestion)

    if suggestion is None:
        # Blind: no suggestion exists, so this is a from-scratch judgement. These are the
        # examples that make anchoring bias measurable, so they are not a fallback.
        print("\n  BLIND EXAMPLE - no suggestion shown. Label from scratch.")
        annotation = _annotate_one(candidate, annotator, pass_number)
        return annotation

    _show_suggestion(suggestion, forced, reasons)
    options = "[C]orrect  [F]lag  [S]kip" if forced else "[A]ccept  [C]orrect  [F]lag  [S]kip"
    options += "   |  inline: i5  i=connectivity  e  sec  ctx  (combine: i5 e)"
    while True:
        choice = _ask(f"\n  {options}: ").strip().lower()
        if choice in ("a", "accept") and not forced:
            return GoldenAnnotation(
                pair_id=candidate.pair_id,
                annotator_id=annotator,
                intent=suggestion.intent,
                security_sensitive=suggestion.security_sensitive,
                context_sufficient=suggestion.context_sufficient,
                should_escalate=suggestion.should_escalate,
                expected_resolution_kind=ExpectedResolutionKind(
                    suggestion.expected_resolution_kind
                ),
                pass_number=pass_number,
                seconds_spent=round(time.time() - started, 1),
                review_action=ReviewAction.ACCEPTED,
                model_suggestion=suggestion.to_dict(),
            )
        if choice in ("a", "accept") and forced:
            print("    one-key accept is disabled here; use C to correct or F to flag")
            continue
        if choice in ("c", "correct"):
            values, changed = _correct(suggestion, list(TAXONOMY.names))
            action = ReviewAction.CORRECTED if changed else ReviewAction.ACCEPTED
            return GoldenAnnotation(
                pair_id=candidate.pair_id,
                annotator_id=annotator,
                intent=values["intent"],
                security_sensitive=values["security_sensitive"],
                context_sufficient=values["context_sufficient"],
                should_escalate=values["should_escalate"],
                expected_resolution_kind=ExpectedResolutionKind(
                    values["expected_resolution_kind"]
                ),
                pass_number=pass_number,
                seconds_spent=round(time.time() - started, 1),
                review_action=action,
                model_suggestion=suggestion.to_dict(),
                corrected_fields=changed,
            )
        if choice in ("f", "flag"):
            reason = _ask("  why does this need deeper review?: ").strip() or "needs review"
            return FlagRecord(
                pair_id=candidate.pair_id,
                annotator_id=annotator,
                reason=reason,
                pass_number=pass_number,
            )
        if choice in ("s", "skip", ""):
            return None
        if choice == "?":
            _show_codebook()
            continue

        try:
            edit = _parse_inline_edit(choice, suggestion, list(TAXONOMY.names))
        except ValueError as exc:
            print(f"    {exc}")
            continue
        if edit is not None:
            values, changed = edit
            # An inline edit that changes nothing is an acceptance, and is recorded as one.
            action = ReviewAction.CORRECTED if changed else ReviewAction.ACCEPTED
            if not changed and forced:
                print("    that edit changes nothing, and one-key accept is disabled here")
                continue
            print(f"    -> {action.value}" + (f", changed {list(changed)}" if changed else ""))
            return GoldenAnnotation(
                pair_id=candidate.pair_id,
                annotator_id=annotator,
                intent=values["intent"],
                security_sensitive=values["security_sensitive"],
                context_sufficient=values["context_sufficient"],
                should_escalate=values["should_escalate"],
                expected_resolution_kind=ExpectedResolutionKind(
                    values["expected_resolution_kind"]
                ),
                pass_number=pass_number,
                seconds_spent=round(time.time() - started, 1),
                review_action=action,
                model_suggestion=suggestion.to_dict(),
                corrected_fields=changed,
            )

        print("    press A, C, F or S, or an inline edit  ('?' for the codebook)")


def _print_status() -> None:
    candidates = read_candidates(CANDIDATES)
    annotations = load_effective_annotations(ANNOTATIONS)
    retractions = read_retractions(ANNOTATIONS)
    stats = coverage(merge(candidates, annotations))
    print(f"Golden set: {CANDIDATES}")
    print(f"  total candidates   {stats['total']}")
    print(f"  HUMAN_LABELED      {stats['human_labelled']}")
    print(f"  UNLABELED          {stats['unlabelled']}")
    print(f"  WEAKLY_LABELED     {stats['weakly_labelled']}  (never permitted as gold)")
    print(f"  MODEL_GENERATED    {stats['model_generated']}  (never permitted as gold)")
    if retractions:
        print(f"  RETRACTED          {len(retractions)}  (record kept, label not counted)")
        for r in retractions:
            print(f"    {r.pair_id}  pass {r.pass_number}  by {r.annotator_id}: {r.reason[:60]}")
    unresolved = unresolved_pair_ids(ANNOTATIONS)
    if unresolved:
        print(f"  FLAGGED UNRESOLVED {len(unresolved)}  (blocks the freeze)")
        for pair_id in unresolved[:10]:
            print(f"    {pair_id}")
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
    parser.add_argument(
        "--retract",
        metavar="PAIR_ID",
        help="withdraw an earlier annotation; the record is kept, the label stops counting",
    )
    parser.add_argument("--reason", help="why the annotation is being withdrawn (required)")
    parser.add_argument(
        "--group",
        choices=("assisted", "blind", "all"),
        default="all",
        help="filter the queue only; the seeded blind/assisted split is never changed",
    )
    parser.add_argument(
        "--assisted",
        action="store_true",
        help="show model pre-annotations for review (SPEC 9.2); blind examples stay blind",
    )
    args = parser.parse_args()

    if args.status:
        _print_status()
        return
    if not args.annotator:
        parser.error("--annotator is required: gold labels must be attributed to a human")

    if args.retract:
        if not args.reason:
            parser.error(
                "--reason is required with --retract: a retraction without a stated reason "
                "is a deletion with extra steps, and the audit trail is the point"
            )
        known = {c.pair_id for c in read_candidates(CANDIDATES)}
        if args.retract not in known:
            parser.error(f"{args.retract!r} is not a candidate in this golden set")
        retraction = Retraction(
            pair_id=args.retract,
            annotator_id=args.annotator,
            reason=args.reason,
            pass_number=args.pass_number,
        )
        append_retraction(ANNOTATIONS, retraction)
        print(f"Retracted {args.retract} (pass {args.pass_number}).")
        print("  The original annotation remains in the log; its label no longer counts.")
        print(f"  reason: {args.reason}")
        print("\nThe example will be offered again on the next annotation run.\n")
        _print_status()
        return

    candidates = read_candidates(CANDIDATES)
    # Effective, so an example whose annotation was retracted is offered again.
    existing = latest_by_pair(
        load_effective_annotations(ANNOTATIONS), pass_number=args.pass_number
    )
    blind = blind_pair_ids(candidates)
    todo = [c for c in _filter_group(candidates, args.group, blind) if c.pair_id not in existing]
    if args.limit:
        todo = todo[: args.limit]

    print(RULE)
    print(f"BLIND ANNOTATION - pass {args.pass_number} - annotator {args.annotator!r}")
    print(RULE)
    print(f"  taxonomy {TAXONOMY.version} ({TAXONOMY.frozen_hash[:16]}...)")
    print(f"  group: {args.group}")
    print(f"  {len(existing)} already done this pass, {len(todo)} to go")
    print("  You will NOT see: the brand's reply, any model prediction, or any suggestion.")
    print("  Guide: docs/ANNOTATION_GUIDE.md   Protocol: docs/GOLDEN_SET.md")
    print("  Enter 's' at the intent prompt to skip, Ctrl-C to stop. Progress is saved as")
    print("  you go, so stopping loses nothing.")

    suggestions = read_suggestions(SUGGESTIONS) if args.assisted else {}
    if args.assisted:
        print(f"  ASSISTED MODE: {len(suggestions)} provisional suggestions available.")
        print("  A suggestion is NOT a label. Nothing is recorded until you accept, correct")
        print("  or flag it, and every action is logged with the suggestion it saw.")

    done = flagged = 0
    for position, candidate in enumerate(todo, start=1):
        forced_left = sum(
            1
            for c in todo[position - 1 :]
            if needs_mandatory_review(suggestions.get(c.pair_id))[0]
        )
        print(_progress_line(position - 1, len(todo), forced_left))
        _show_candidate(candidate, position, len(todo))
        try:
            if args.assisted:
                outcome = _review_one(
                    candidate,
                    suggestions.get(candidate.pair_id),
                    args.annotator,
                    args.pass_number,
                )
            else:
                outcome = _annotate_one(candidate, args.annotator, args.pass_number)
        except KeyboardInterrupt:
            print("\n\nStopped. Everything answered so far is saved.")
            break
        if outcome is None:
            print("  skipped.")
            continue
        if isinstance(outcome, FlagRecord):
            append_flag(ANNOTATIONS, outcome)
            flagged += 1
            print("  flagged for deeper review; it stays unresolved and blocks the freeze.")
            continue
        append_annotation(ANNOTATIONS, outcome)
        done += 1
        print(f"  saved [{outcome.review_action.value}] ({outcome.seconds_spent:.0f}s).")

    print(f"\n{done} annotation(s) and {flagged} flag(s) written to {ANNOTATIONS}")
    _print_status()


if __name__ == "__main__":
    main()
