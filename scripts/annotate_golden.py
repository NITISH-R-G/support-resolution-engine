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

from hiver_support.golden import paths  # noqa: E402
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
    MIN_CONFIDENT,
    ModelSuggestion,
    blind_pair_ids,
    needs_mandatory_review,
    read_suggestions,
)
from hiver_support.taxonomy import TAXONOMY  # noqa: E402

GOLDEN_DIR = ROOT / "data" / "golden"
# Full text is rebuilt locally (scripts/materialize_text.py --golden); the committed candidates.jsonl
# is text-free. v1 is the annotated set.
CANDIDATES = paths.local_candidates("v1")
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


# Plain-English names for the person annotating. Keyed by the frozen taxonomy names, which stay
# the only values ever saved. Tests assert the keys match the taxonomy exactly, so a label can
# neither drift from the codebook nor add a category to it.
INTENT_LABELS = {
    "device_malfunction": "Device malfunction",
    "battery_charging": "Battery & charging",
    "apps_and_services": "Apps & services",
    "billing_and_subscription": "Billing & subscription",
    "connectivity": "Connectivity",
    "howto_information": "How-to / information",
    "complaint_feedback": "Complaint / feedback",
    "account_access": "Account access",
    "repair_order_replacement": "Repair / order / replacement",
    "other_unclear": "Other / unclear",
}
RESOLUTION_LABELS = {
    "self_serve_steps": "Self-serve steps",
    "information": "Information",
    "human_action_required": "Human action required",
    "no_resolution_possible": "No resolution possible",
    "unclear": "Unclear",
}
CONFIDENCE_LABELS = {"high": "Very sure", "medium": "Fairly sure", "low": "Not sure"}
ATTRIBUTE_LABELS = {
    "security_sensitive": "Security-sensitive",
    "context_sufficient": "Enough context",
}
FIELD_ORDER = (
    "intent",
    "security_sensitive",
    "context_sufficient",
    "should_escalate",
    "expected_resolution_kind",
)


def _yes_no(value: bool) -> str:
    return "Yes" if value else "No"


def _readable(text: str) -> str:
    """Codebook prose with identifiers softened, so the screen never shows snake_case."""
    return _safe(text).replace("_", " ")


def _hint(intent) -> str:
    """First sentence of the frozen definition, shortened to one line."""
    text = intent.definition.split(". ")[0]
    if len(text) > 64:
        text = text[:64].rsplit(" ", 1)[0] + "..."
    return _readable(text)


def _ask_flag(prompt: str, default: bool) -> bool:
    print(f"\n  {prompt}")
    print("    [Y] Yes")
    print("    [N] No")
    while True:
        answer = _ask(f"  Your choice (Enter = {_yes_no(default)}): ").lower()
        if not answer:
            return default
        if answer in ("y", "yes"):
            return True
        if answer in ("n", "no"):
            return False
        print("    Please type Y or N.")


def _ask_choice(prompt: str, options: list[str], default: str | None = None) -> str:
    while True:
        answer = _ask(f"  {prompt}: ").strip()
        if not answer and default is not None:
            return default
        if answer.isdigit() and 1 <= int(answer) <= len(options):
            return options[int(answer) - 1]
        if answer in options:
            return answer
        print("    Please type one of the numbers shown.")


def _show_intents(title: str = "What is the intent?", extra: str = "") -> None:
    print(f"\n  {title}{extra}")
    for index, intent in enumerate(TAXONOMY.intents, start=1):
        print(f"    {index:>2}. {INTENT_LABELS[intent.name]:<30} {_hint(intent)}")
    print("    [?] Show full definitions")


def _show_resolutions(title: str = "What kind of resolution is expected?", extra: str = "") -> None:
    print(f"\n  {title}{extra}")
    for index, kind in enumerate(RESOLUTION_KINDS, start=1):
        print(f"    {index}. {RESOLUTION_LABELS[kind.value]}")


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
        flag = "  (usually handled by a person)" if intent.escalation_sensitive else ""
        print(f"\n  {index}. {INTENT_LABELS[intent.name]}{flag}")
        print(f"     {_readable(intent.definition)}")
        if intent.includes:
            print(f"     Includes: {_readable('; '.join(intent.includes))}")
        if intent.excludes:
            print(f"     Does not include: {_readable('; '.join(intent.excludes))}")
        if intent.confusions:
            others = ", ".join(INTENT_LABELS[c] for c in intent.confusions if c in INTENT_LABELS)
            print(f"     Often confused with: {others}")
    print("\n  ATTRIBUTES (independent of the intent - a message can be any intent AND these)")
    for attribute in TAXONOMY.attributes:
        print(f"\n  {ATTRIBUTE_LABELS.get(attribute.name, _readable(attribute.name))}")
        print(f"     {_readable(attribute.definition)}")
        print(f"     Rule: {_readable(attribute.annotation_rule)}")
    print("\n  Full guide: docs/ANNOTATION_GUIDE.md")
    print(RULE)


def _show_candidate(candidate, position: int, total: int) -> None:
    print("\n" + RULE)
    print(f"Example {position} of {total}   ({candidate.created_at:%d %b %Y})")
    print(RULE)
    if candidate.context:
        print("\n  EARLIER IN THIS CONVERSATION")
        for turn in candidate.context:
            who = "Customer" if turn.author_role == "customer" else "Apple Support"
            print(f"    {who}: {_safe(turn.text)[:300]}")
    print("\n  CUSTOMER MESSAGE")
    for line in _safe(candidate.customer_message).splitlines() or [""]:
        print(f"    {line}")


def _ask_intent(names: list[str]) -> str:
    """Intent by menu number or frozen name; 's' skips, '?' shows definitions."""
    while True:
        answer = _ask("  Your choice: ").strip()
        if answer.lower() in ("s", "skip"):
            return "s"
        if answer == "?":
            return "?"
        if answer.isdigit() and 1 <= int(answer) <= len(names):
            return names[int(answer) - 1]
        if answer in names:
            return answer
        print("    Please type one of the numbers shown.")


def _annotate_one(candidate, annotator: str, pass_number: int) -> GoldenAnnotation | None:
    """Label one example from scratch. Used for blind examples: no suggestion is shown."""
    started = time.time()
    names = list(TAXONOMY.names)
    while True:
        _show_intents()
        print("    [S] Skip this one for now")
        intent = _ask_intent(names)
        if intent != "?":
            break
        _show_codebook()
        # Reading the definitions is not deciding, so it does not count towards the duration.
        started = time.time()
    if intent == "s":
        return None

    security = _ask_flag("Is this security-sensitive?", default=False)
    context_ok = _ask_flag(
        "Is there enough context to understand the customer's issue?", default=True
    )
    escalate = _ask_flag("Should this be escalated to a human?", default=False)

    _show_resolutions()
    kind = _ask_choice("Your choice", [k.value for k in RESOLUTION_KINDS])

    ambiguous = _ask_flag("Could this reasonably be a different intent as well?", default=False)
    alternative = None
    if ambiguous:
        _show_intents("Which other intent could it be?")
        print(f"    {len(names) + 1:>2}. None of these")
        answer = _ask_choice("Your choice (Enter = none)", names + ["none"], default="none")
        alternative = None if answer in ("none", intent) else answer

    print("\n  How sure are you about this label?")
    for index, level in enumerate(LabelConfidence, start=1):
        print(f"    {index}. {CONFIDENCE_LABELS[level.value]}")
    confidence = _ask_choice(
        "Your choice (Enter = Very sure)",
        [c.value for c in LabelConfidence],
        default=LabelConfidence.HIGH.value,
    )
    reference = _ask("  Optional - what should a good reply say? (Enter to skip): ")
    notes = _ask("  Optional notes (Enter to skip): ")

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


def _filter_group(candidates, group: str, blind: set[str]):
    """Filter the annotation queue only. The split itself is fixed by ``blind_pair_ids``."""
    if group == "assisted":
        return [c for c in candidates if c.pair_id not in blind]
    if group == "blind":
        return [c for c in candidates if c.pair_id in blind]
    return list(candidates)


def _progress_line(done: int, total: int, forced_left: int) -> str:
    remaining = max(total - done, 0)
    return (
        f"  Progress: {done} of {total} done, {remaining} left "
        f"({forced_left} need a careful look)"
    )


def _plain_reasons(suggestion: ModelSuggestion) -> list[str]:
    """Why a suggestion needs a careful look, in words rather than thresholds."""
    reasons = []
    if suggestion.confidence < MIN_CONFIDENT:
        reasons.append("the model was not confident about it")
    if suggestion.security_sensitive:
        reasons.append("the model thinks it may involve account or device security")
    if not suggestion.context_sufficient:
        reasons.append("the model thinks the message may be too unclear to act on")
    if TAXONOMY.get(suggestion.intent).escalation_sensitive:
        label = INTENT_LABELS[suggestion.intent].lower()
        reasons.append(f"it is about {label}, which is normally handled by a person")
    if suggestion.should_escalate:
        reasons.append("the model suggests handing it to a human")
    return reasons or ["it was marked for a careful look"]


def _show_suggestion(suggestion: ModelSuggestion, forced: bool) -> None:
    """The suggestion in plain English, unmistakably a guess rather than a label."""
    print("\n  MODEL'S SUGGESTION  (a model's guess - not a label until you confirm it)")
    print(f"    Intent: {INTENT_LABELS[suggestion.intent]}")
    print(f"    Security sensitive: {_yes_no(suggestion.security_sensitive)}")
    print(f"    Context sufficient: {_yes_no(suggestion.context_sufficient)}")
    print(f"    Should escalate: {_yes_no(suggestion.should_escalate)}")
    print(f"    Expected resolution: {RESOLUTION_LABELS[suggestion.expected_resolution_kind]}")
    if suggestion.rationale:
        print(f"    Why: {_readable(suggestion.rationale)}")
    if forced:
        print("\n  Please take a careful look at this one, because:")
        for reason in _plain_reasons(suggestion):
            print(f"    - {reason}")


def _ask_keep(options: list[str], current: str) -> str:
    """A numbered choice where Enter keeps the model's answer."""
    while True:
        answer = _ask("  Your choice: ").strip()
        if not answer:
            return current
        if answer == "?":
            _show_codebook()
            continue
        if answer.isdigit() and 1 <= int(answer) <= len(options):
            return options[int(answer) - 1]
        if answer in options:
            return answer
        print("    Please type one of the numbers shown, or press Enter to keep it.")


def _correct(suggestion: ModelSuggestion) -> tuple[dict, tuple[str, ...]]:
    """Walk the five fields in plain English. Enter keeps the model's answer for any of them."""
    values = {field: getattr(suggestion, field) for field in FIELD_ORDER}
    print("\n  Let's correct it. For each question, press Enter to keep the model's answer.")

    _show_intents(
        "What is the correct intent?",
        f"   (Enter keeps: {INTENT_LABELS[values['intent']]})",
    )
    values["intent"] = _ask_keep(list(TAXONOMY.names), values["intent"])

    for field, question in (
        ("security_sensitive", "Is this security-sensitive?"),
        ("context_sufficient", "Is there enough context to understand the customer's issue?"),
        ("should_escalate", "Should this be escalated to a human?"),
    ):
        values[field] = _ask_flag(question, default=values[field])

    _show_resolutions(
        "What kind of resolution is expected?",
        f"   (Enter keeps: {RESOLUTION_LABELS[values['expected_resolution_kind']]})",
    )
    values["expected_resolution_kind"] = _ask_keep(
        [k.value for k in RESOLUTION_KINDS], values["expected_resolution_kind"]
    )

    changed = tuple(f for f in FIELD_ORDER if values[f] != getattr(suggestion, f))
    return values, changed


def _reviewed_label(candidate, suggestion, values, changed, annotator, pass_number, started):
    """A human decision about a suggestion. ACCEPTED when nothing changed, CORRECTED otherwise."""
    return GoldenAnnotation(
        pair_id=candidate.pair_id,
        annotator_id=annotator,
        intent=values["intent"],
        security_sensitive=values["security_sensitive"],
        context_sufficient=values["context_sufficient"],
        should_escalate=values["should_escalate"],
        expected_resolution_kind=ExpectedResolutionKind(values["expected_resolution_kind"]),
        pass_number=pass_number,
        seconds_spent=round(time.time() - started, 1),
        review_action=ReviewAction.CORRECTED if changed else ReviewAction.ACCEPTED,
        model_suggestion=suggestion.to_dict(),
        corrected_fields=changed,
    )


def _review_one(candidate, suggestion, annotator: str, pass_number: int):
    """Review one example. Returns an annotation, a FlagRecord, or None to skip."""
    started = time.time()
    if suggestion is None:
        # Blind: nothing to review, so the example is labelled from scratch. These are what
        # make anchoring bias measurable, so no suggestion may ever be shown here.
        print("\n  This one has no model suggestion - please answer the questions yourself.")
        return _annotate_one(candidate, annotator, pass_number)

    forced, _ = needs_mandatory_review(suggestion)
    _show_suggestion(suggestion, forced)
    while True:
        print("\n  Is this suggestion correct?")
        print("    [Y] Yes")
        print("    [N] No, I want to correct it")
        print("    [S] Skip")
        print("    [F] Flag")
        print("    [?] Show full definitions")
        choice = _ask("  Your choice: ").strip().lower()

        if choice in ("y", "yes"):
            if forced:
                # The careful-look gate, kept as one plain confirmation.
                print("\n  Before saving: are you sure everything in the suggestion is correct?")
                print("    [Y] Yes, it is correct")
                print("    [N] No, go back")
                if _ask("  Your choice: ").strip().lower() not in ("y", "yes"):
                    continue
            values = {field: getattr(suggestion, field) for field in FIELD_ORDER}
            return _reviewed_label(
                candidate, suggestion, values, (), annotator, pass_number, started
            )
        if choice in ("n", "no"):
            values, changed = _correct(suggestion)
            return _reviewed_label(
                candidate, suggestion, values, changed, annotator, pass_number, started
            )
        if choice in ("s", "skip"):
            return None
        if choice in ("f", "flag"):
            reason = _ask("  In a few words, what makes this one hard to decide?: ").strip()
            return FlagRecord(
                pair_id=candidate.pair_id,
                annotator_id=annotator,
                reason=reason or "needs a closer look",
                pass_number=pass_number,
            )
        if choice == "?":
            _show_codebook()
            continue
        print("    Please type Y, N, S or F.")


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
    if args.assisted:
        print("  Suggestions are a model's guesses. Nothing counts until you confirm or correct it.")
    else:
        print("  No model suggestions are shown in this mode.")
    print("  Guide: docs/ANNOTATION_GUIDE.md   Protocol: docs/GOLDEN_SET.md")
    print("  Press Ctrl-C to stop at any time - everything answered so far is saved.")

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
            print("  Flagged - it stays open until you come back to it.")
            continue
        append_annotation(ANNOTATIONS, outcome)
        done += 1
        print(f"  Saved ({outcome.review_action.value}).")

    print(f"\n{done} annotation(s) and {flagged} flag(s) written to {ANNOTATIONS}")
    _print_status()


if __name__ == "__main__":
    main()
