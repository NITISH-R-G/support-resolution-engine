"""The AppleSupport intent taxonomy.

**Status: FROZEN at v0.3.0** (hash `613f5dfec125...`), approved 2026-09-10 after three review
rounds. Post-freeze changes require a NEW version and a decision-log entry; model performance
may never motivate a change (`DECISION_LOG.md` D14), and any change after golden-set labelling
begins invalidates the golden set.

Ten intents carry the request; two orthogonal attributes carry safety and context. Safety is
an attribute rather than a label because 306 of 340 security-sensitive messages (90%) in the
train split fell outside the account topic — an account-shaped security intent would have
captured a tenth of the safety signal.

Every label here was derived from the real AppleSupport train split and is defended in
`docs/INTENT_TAXONOMY.md` against evidence in `reports/taxonomy_discovery.json` (clustering)
and `reports/taxonomy_probes.json` (hypothesis probes). Nothing is inherited from another
implementation, and no label exists because it improves a metric — no model has been trained
and no evaluation has been run.

Two properties of the data shaped this design more than anything else:

1. **The corpus is dominated by one event.** Embedding clusters at k=12 and k=16 are mostly
   variants of "the iOS 11 update broke my phone". Clustering surfaces *topic of the moment*,
   not *support intent*, so labels are defined by the support action a message calls for, not
   by the vocabulary clusters happened to isolate.
2. **Prevalence figures are FLOORS.** They come from high-precision, low-recall lexical
   probes; 45.7% of the split matches no probe at all. True prevalence is unknown until the
   golden set is hand-labelled, and is deliberately not estimated here.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field, replace

VERSION = "0.3.0"


class TaxonomyError(ValueError):
    """Raised when a taxonomy is structurally invalid or misused."""


@dataclass(frozen=True, slots=True)
class IntentExample:
    """A real corpus message, PII-masked, with its pair id for traceability."""

    pair_id: str
    text: str
    note: str


@dataclass(frozen=True, slots=True)
class TieBreak:
    """Which label wins when a message matches two, and why."""

    winner: str
    loser: str
    rule: str


@dataclass(frozen=True, slots=True)
class Intent:
    """One label.

    ``prevalence_floor`` is a **floor**, not an estimate: it is the share of the train split
    matched by deliberately conservative lexical probes, which undercount substantially.
    True prevalence is unknown until the golden set is hand-labelled.
    """

    name: str
    definition: str
    includes: tuple[str, ...]
    excludes: tuple[str, ...]
    examples: tuple[IntentExample, ...]
    escalation_sensitive: bool
    auto_handleable: bool
    prevalence_floor: float
    confusions: tuple[str, ...]
    why_intent: str = ""
    is_catch_all: bool = False
    requires_context: bool = False


@dataclass(frozen=True, slots=True)
class Attribute:
    """An orthogonal axis annotated alongside the intent, never instead of it.

    Safety and context are not kinds of request. Encoding safety as an intent was measured to
    lose most of the signal: 306 of 340 security-sensitive messages in the train split (90%)
    fell outside the account topic, so an account-shaped security label would have captured a
    tenth of the safety-relevant traffic and let the rest be labelled by topic with the flag
    silently dropped.
    """

    name: str
    definition: str
    routing_consequence: str
    forces_escalation: bool
    annotation_rule: str


@dataclass(frozen=True, slots=True)
class Taxonomy:
    version: str
    intents: tuple[Intent, ...]
    tie_breaks: tuple[TieBreak, ...]
    frozen: bool
    attributes: tuple[Attribute, ...] = ()
    provenance: dict = field(default_factory=dict)

    @property
    def frozen_hash(self) -> str:
        """Content hash recorded at freeze time.

        Equal to ``content_hash()`` by construction: freezing records that *this* content was
        approved, so the two must never diverge. A test asserts the equality, which is what
        makes a silent post-freeze edit detectable.
        """
        return self.content_hash()

    def must_escalate(
        self,
        intent: str,
        *,
        security_sensitive: bool = False,
        context_sufficient: bool = True,
    ) -> bool:
        """Whether this combination must escalate, per SPEC section 8.

        Attributes dominate the intent: ``security_sensitive`` forces escalation for *every*
        intent, which is the entire reason safety is an axis rather than a label.
        """
        if intent not in {i.name for i in self.intents}:
            raise TaxonomyError(f"unknown intent: {intent!r}")
        if security_sensitive or not context_sufficient:
            return True
        return next(i for i in self.intents if i.name == intent).escalation_sensitive

    def __post_init__(self) -> None:
        names = [i.name for i in self.intents]
        duplicates = {n for n in names if names.count(n) > 1}
        if duplicates:
            raise TaxonomyError(f"duplicate intent names: {sorted(duplicates)}")

        known = set(names)
        for rule in self.tie_breaks:
            if rule.winner not in known or rule.loser not in known:
                raise TaxonomyError(
                    f"tie-break references unknown label: {rule.winner} > {rule.loser}"
                )
            if rule.winner == rule.loser:
                raise TaxonomyError(f"tie-break is self-referential: {rule.winner}")

        self._assert_acyclic()

    def _assert_acyclic(self) -> None:
        """A cycle in the precedence graph would make resolution non-deterministic."""
        edges: dict[str, set[str]] = {}
        for rule in self.tie_breaks:
            edges.setdefault(rule.winner, set()).add(rule.loser)

        def reaches(start: str, target: str, seen: set[str]) -> bool:
            if start in seen:
                return False
            seen.add(start)
            for node in edges.get(start, ()):
                if node == target or reaches(node, target, seen):
                    return True
            return False

        for rule in self.tie_breaks:
            if reaches(rule.loser, rule.winner, set()):
                raise TaxonomyError(
                    f"cycle in tie-break precedence: {rule.winner} > {rule.loser} > ... > "
                    f"{rule.winner}"
                )

    # ------------------------------------------------------------------ lookups

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(i.name for i in self.intents)

    @property
    def catch_all(self) -> Intent:
        for intent in self.intents:
            if intent.is_catch_all:
                return intent
        raise TaxonomyError("taxonomy has no catch-all label")

    def get(self, name: str) -> Intent:
        for intent in self.intents:
            if intent.name == name:
                return intent
        raise TaxonomyError(f"unknown intent: {name!r}")

    # ------------------------------------------------------------------ resolution

    def resolve(self, candidates: list[str]) -> str:
        """Reduce a multi-intent candidate set to one label.

        Applies documented tie-breaks; where no rule exists, falls back to declaration
        order, which is arranged so the more escalation-sensitive and more specific labels
        come first. The fallback exists so resolution is always *deterministic* — an
        undocumented pair must never resolve differently depending on argument order.
        """
        if not candidates:
            return self.catch_all.name

        unknown = [c for c in candidates if c not in set(self.names)]
        if unknown:
            raise TaxonomyError(f"unknown intent(s): {unknown}")

        unique = list(dict.fromkeys(candidates))
        if len(unique) == 1:
            return unique[0]

        precedence = {name: index for index, name in enumerate(self.names)}
        beats = {(r.winner, r.loser) for r in self.tie_breaks}

        best = unique[0]
        for candidate in unique[1:]:
            if (candidate, best) in beats:
                best = candidate
            elif (best, candidate) in beats:
                continue
            elif precedence[candidate] < precedence[best]:
                best = candidate
        return best

    # ------------------------------------------------------------------ versioning

    def content_hash(self) -> str:
        """Stable hash over content, excluding the freeze flag.

        Freezing must not change the hash: it records that the *same* content was approved.
        """
        payload = json.dumps(
            {
                "version": self.version,
                "intents": [asdict(i) for i in self.intents],
                "tie_breaks": [asdict(t) for t in self.tie_breaks],
                "attributes": [asdict(a) for a in self.attributes],
            },
            sort_keys=True,
            ensure_ascii=False,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def freeze(self) -> Taxonomy:
        if self.frozen:
            raise TaxonomyError("taxonomy is already frozen; bump the version instead")
        return replace(self, frozen=True)

    def to_dict(self) -> dict:
        return {
            "version": self.version,
            "frozen": self.frozen,
            "content_hash": self.content_hash(),
            "intents": [asdict(i) for i in self.intents],
            "tie_breaks": [asdict(t) for t in self.tie_breaks],
            "attributes": [asdict(a) for a in self.attributes],
            "provenance": self.provenance,
        }

    @classmethod
    def from_dict(cls, payload: dict) -> Taxonomy:
        intents = tuple(
            Intent(
                **{
                    **intent,
                    "includes": tuple(intent["includes"]),
                    "excludes": tuple(intent["excludes"]),
                    "confusions": tuple(intent["confusions"]),
                    "examples": tuple(IntentExample(**e) for e in intent["examples"]),
                }
            )
            for intent in payload["intents"]
        )
        return cls(
            version=payload["version"],
            intents=intents,
            tie_breaks=tuple(TieBreak(**t) for t in payload["tie_breaks"]),
            frozen=payload.get("frozen", False),
            attributes=tuple(Attribute(**a) for a in payload.get("attributes", ())),
            provenance=payload.get("provenance", {}),
        )


def _ex(pair_id: str, text: str, note: str) -> IntentExample:
    return IntentExample(pair_id=pair_id, text=text, note=note)


# ---------------------------------------------------------------------------------------
# Candidate taxonomy. Declaration order IS the fallback precedence: escalation-sensitive and
# account-specific labels first, generic device complaints later, catch-all last.
# ---------------------------------------------------------------------------------------

_INTENTS: tuple[Intent, ...] = (
    Intent(
        name="account_access",
        definition=(
            "Getting into, or staying signed in to, an Apple ID or account: sign-in "
            "failures, password and passcode resets, two-factor problems and lockouts. "
            "Suspected compromise is NOT a separate intent: it is this intent with "
            "security_sensitive=true."
        ),
        includes=(
            "cannot sign in to Apple ID; forgotten password or passcode",
            "two-factor / verification-code problems; account disabled or locked",
            "suspected compromise or unauthorised access (also set security_sensitive)",
        ),
        excludes=(
            "a Wi-Fi network password — that is connectivity",
            "a disputed charge with no access problem — that is billing_and_subscription",
            "device passcode forgotten because the device is broken — device_malfunction",
        ),
        examples=(
            _ex(
                "1315478__1315475",
                '[tweet-text redacted: tweet_id=1315478 sha256=98244605aebdaa0f]',
                "credential reset; identity verification needed before any action",
            ),
            _ex(
                "1122188__1122187",
                '[tweet-text redacted: tweet_id=1122188 sha256=953bc48b5bed7e8a]',
                "phishing verification; always escalate, never auto-confirm legitimacy",
            ),
            _ex(
                "1714094__1714093",
                '[tweet-text redacted: tweet_id=1714094 sha256=1fd410c92b6116b5]',
                "update-attributed but the blocked action is account access",
            ),
        ),
        escalation_sensitive=False,
        auto_handleable=True,
        prevalence_floor=0.0158,
        why_intent=(
            "An intent rather than a safety attribute: the resolution evidence is the account- "
            "recovery procedure, which exists independently of whether a security concern is "
            "present. Safety is carried by the orthogonal security_sensitive attribute precisely so "
            "a forgotten password and a suspected takeover share this resolution path while "
            "differing in escalation. "
        ),
        confusions=("connectivity", "billing_and_subscription", "howto_information"),
    ),
    Intent(
        name="billing_and_subscription",
        definition=(
            "Money: charges, refunds, receipts and invoices, plus the lifecycle of a paid "
            "subscription such as Apple Music or iCloud storage."
        ),
        includes=(
            "unexpected, duplicated or disputed charges; refund requests",
            "subscription start, renewal, price change or cancellation",
            "free-trial and auto-renewal questions; payment method failures",
        ),
        excludes=(
            "'charged my phone' meaning the battery — that is battery_charging",
            "the price of a device or repair before purchase — howto_information",
            "a fraudulent charge where the account itself is compromised — account_security",
        ),
        examples=(
            _ex(
                "810181__810179",
                '[tweet-text redacted: tweet_id=810181 sha256=d51c3ebff09e5bb7]'
                "disputed charge plus refund demand; account-specific financial action",
            ),
            _ex(
                "1447244__1447242",
                '[tweet-text redacted: tweet_id=1447244 sha256=011fccd019469a4d]',
                "subscription billing dispute",
            ),
            _ex(
                "2042527__2042526",
                '[tweet-text redacted: tweet_id=2042527 sha256=84c54608f720a72c]',
                "refund demand arising from an update complaint; money wins the tie-break",
            ),
        ),
        escalation_sensitive=True,
        auto_handleable=False,
        prevalence_floor=0.018,
        why_intent=(
            "An intent on escalation-policy grounds: money is a hard-rule escalation category in "
            "SPEC section 8 regardless of channel behaviour. The resolution evidence is charge "
            "explanation, refund policy and subscription lifecycle, none of which appear in "
            "troubleshooting cases. "
        ),
        confusions=("battery_charging", "account_access", "repair_order_replacement"),
    ),
    Intent(
        name="repair_order_replacement",
        definition=(
            "Physical logistics: repair, warranty and AppleCare, hardware replacement, and "
            "the status of a purchase, order or delivery."
        ),
        includes=(
            "repair bookings, Genius Bar, service centres, out-of-warranty questions",
            "physical damage: cracked screen, liquid damage, dead hardware",
            "order status, shipping, delivery and returns",
        ),
        excludes=(
            "software 'replacement' such as a replacement keyboard app — apps_and_services",
            "the cost of a product before any order exists — howto_information",
            "a battery that drains but is not being replaced — battery_charging",
        ),
        examples=(
            _ex(
                "972534__972533",
                '[tweet-text redacted: tweet_id=972534 sha256=c00c939ca098a4e6]',
                "physical damage plus repair availability",
            ),
            _ex(
                "1292114__1292113",
                '[tweet-text redacted: tweet_id=1292114 sha256=e3fbf7c136f28633]',
                "warranty/upgrade-programme eligibility with physical damage",
            ),
            _ex(
                "939563__939565",
                '[tweet-text redacted: tweet_id=939563 sha256=ed7b42479fe77ef1]',
                "multi-intent: update attribution plus repair history",
            ),
        ),
        escalation_sensitive=True,
        auto_handleable=False,
        prevalence_floor=0.011,
        why_intent=(
            "The most strongly evidenced intent boundary in the corpus: the support action is "
            "logistics routing rather than resolution, measured at 0.221 to 0.288 maximum handling "
            "difference against every other label. The resolution evidence is appointment, warranty "
            "and order status, and escalation applies because physical goods and money are "
            "involved. "
        ),
        confusions=("device_malfunction", "howto_information", "billing_and_subscription"),
    ),
    Intent(
        name="connectivity",
        definition=(
            "The device cannot connect or stay connected: Wi-Fi, Bluetooth, cellular data, "
            "hotspot, AirDrop or pairing between devices."
        ),
        includes=(
            "Wi-Fi will not join, drops, or rejects a network password",
            "Bluetooth pairing failures; toggles turning themselves back on",
            "no cellular service, mobile data or hotspot failures",
        ),
        excludes=(
            "an Apple ID sign-in failure — that is account_security",
            "an app failing for reasons unrelated to the network — apps_and_services",
        ),
        examples=(
            _ex(
                "2054700__2054698",
                '[tweet-text redacted: tweet_id=2054700 sha256=1d904f557e0eed75]',
                "the iOS 11 Wi-Fi/Bluetooth toggle behaviour; update-attributed",
            ),
            _ex(
                "1217580__1217578",
                '[tweet-text redacted: tweet_id=1217580 sha256=29c3679f46d3cfa8]',
                "connectivity contrasted across OS versions",
            ),
            _ex(
                "52265__52263",
                '[tweet-text redacted: tweet_id=52265 sha256=76adb7c42288ed13]',
                "contains 'password' but is connectivity, not account access",
            ),
        ),
        escalation_sensitive=False,
        auto_handleable=True,
        prevalence_floor=0.029,
        why_intent=(
            "An intent rather than a symptom of device_malfunction: the resolution evidence is "
            "network-specific, covering network-reset procedures and carrier or router interaction, "
            "and retrieval must surface connectivity cases. The support action follows a different "
            "diagnostic path from hardware failure. "
        ),
        confusions=("account_access", "device_malfunction"),
    ),
    Intent(
        name="battery_charging",
        definition=(
            "Battery and power behaviour: drain rate, charge percentage, charging failures, "
            "and heat associated with charging or power use."
        ),
        includes=(
            "battery draining faster than expected; sudden percentage drops",
            "device will not charge, or will not hold charge; charger not working",
            "overheating during use or charging",
        ),
        excludes=(
            "'charged' meaning a financial charge — billing_and_subscription",
            "a swollen or physically damaged battery needing replacement — "
            "repair_order_replacement",
        ),
        examples=(
            _ex(
                "195904__195906",
                '[tweet-text redacted: tweet_id=195904 sha256=07815454e5771f87]',
                "battery drain, mid-thread but self-describing",
            ),
            _ex(
                "2080925__2080924",
                '[tweet-text redacted: tweet_id=2080925 sha256=1efeab0360c9a039]',
                "battery symptom attributed to an update; symptom wins the tie-break",
            ),
            _ex(
                "259429__259428",
                '[tweet-text redacted: tweet_id=259429 sha256=ac3cbc770a325179]',
                "battery plus hostile complaint tone",
            ),
        ),
        escalation_sensitive=False,
        auto_handleable=True,
        prevalence_floor=0.073,
        why_intent=(
            "An intent rather than a topic slice of device_malfunction: the resolution evidence is "
            "battery-specific, covering battery-health readings, charge-cycle guidance and "
            "charging-hardware checks, and is not interchangeable with general device diagnostics. "
            "Retrieval on battery cases surfaces a distinct evidence type. "
        ),
        confusions=("billing_and_subscription", "device_malfunction"),
    ),
    Intent(
        name="apps_and_services",
        definition=(
            "An Apple app or service misbehaves: Apple Music, iTunes, iCloud, iMessage, "
            "FaceTime, App Store, Siri, Safari, Photos, Apple Pay."
        ),
        includes=(
            "a named Apple app crashing, hanging or refusing to open",
            "iCloud sync or storage behaviour; iMessage or FaceTime not delivering",
            "App Store or iTunes purchase and download failures",
        ),
        excludes=(
            "the whole device is unusable, not one app — device_malfunction",
            "a charge for a service — billing_and_subscription",
            "a third-party app, which Apple support does not own",
        ),
        examples=(
            _ex(
                "977661__977660",
                '[tweet-text redacted: tweet_id=977661 sha256=ff1a811ac41f77ac]',
                "single named app misbehaving",
            ),
            _ex(
                "1329303__1329305",
                '[tweet-text redacted: tweet_id=1329303 sha256=67c167e24c252ef5]',
                "service-triggered freeze; boundary with device_malfunction",
            ),
            _ex(
                "2008786__2008785",
                '[tweet-text redacted: tweet_id=2008786 sha256=0f8d5006db3be8f5]',
                "service complaint with no explicit request",
            ),
        ),
        escalation_sensitive=False,
        auto_handleable=True,
        prevalence_floor=0.047,
        why_intent=(
            "An intent rather than a product-attribution label: the resolution evidence is per- "
            "service guidance tied to a named Apple service, and retrieval must be scoped to that "
            "service. The support action differs from device diagnostics because the fault lies in "
            "a service the customer does not control. "
        ),
        confusions=("device_malfunction", "complaint_feedback"),
    ),
    Intent(
        name="device_malfunction",
        definition=(
            "The device or one of its built-in functions is not working correctly, with no "
            "update attribution and no more specific category."
        ),
        includes=(
            "freezing, crashing, unresponsive screen, restarts",
            "hardware controls misbehaving: buttons, switches, speakers, camera",
            "general 'my phone is broken' with symptoms but no attributed cause",
        ),
        excludes=(
            "an update is blamed — still classify by the request, not the attribution",
            "the symptom is battery, connectivity or a named app — those labels",
            "the device is physically damaged and needs service — repair_order_replacement",
        ),
        examples=(
            _ex(
                "983881__983882",
                '[tweet-text redacted: tweet_id=983881 sha256=ae8a252d0cf6ec43]',
                "stuck operation with no update attribution",
            ),
            _ex(
                "1704621__1704620",
                '[tweet-text redacted: tweet_id=1704621 sha256=4f8f426da0bbd6de]',
                "symptom described, cause unattributed",
            ),
            _ex(
                "229207__229206",
                '[tweet-text redacted: tweet_id=229207 sha256=568e450d424a4b0c]',
                "boundary case with apps_and_services",
            ),
        ),
        escalation_sensitive=False,
        auto_handleable=True,
        prevalence_floor=0.059,
        why_intent=(
            "An intent rather than a symptom bucket: the required resolution evidence is a "
            "diagnostic troubleshooting sequence for hardware or OS-level failure, and retrieval "
            "must surface prior cases with the same failure mode. The support action differs from "
            "an information request, which needs documentation rather than diagnosis. "
        ),
        confusions=("apps_and_services", "connectivity", "repair_order_replacement"),
    ),
    Intent(
        name="howto_information",
        definition=(
            "A request for instructions, capability or policy information, where nothing is "
            "reported as broken."
        ),
        includes=(
            "'how do I…', 'is there a way to…', 'where do I find…'",
            "whether a feature or device supports something",
            "pricing or availability of a product or service before any purchase",
        ),
        excludes=(
            "'how do I fix this fault' — the fault decides the label",
            "an order that already exists — repair_order_replacement",
        ),
        examples=(
            _ex(
                "1350656__1350654",
                '[tweet-text redacted: tweet_id=1350656 sha256=80296360e36b8ce8]',
                "pure capability question, nothing broken",
            ),
            _ex(
                "1228768__1228767",
                '[tweet-text redacted: tweet_id=1228768 sha256=aea44880fc269b3f]',
                "capability question that brushes billing but requests no charge",
            ),
            _ex(
                "779492__779490",
                '[tweet-text redacted: tweet_id=779492 sha256=3be5efcbd3f2b378]',
                "explicit how-to",
            ),
        ),
        escalation_sensitive=False,
        auto_handleable=True,
        prevalence_floor=0.020,
        why_intent=(
            "An intent rather than a wording variant: no fault is reported, so the support action "
            "is to supply documentation rather than to diagnose. The resolution evidence is an "
            "article or feature explanation, a different evidence type from any troubleshooting "
            "case, and auto-handling is materially safer than where a fault exists. "
        ),
        confusions=("repair_order_replacement", "device_malfunction"),
    ),
    Intent(
        name="complaint_feedback",
        definition=(
            "Expressive dissatisfaction, praise, or product feedback carrying no request "
            "that support could act on."
        ),
        includes=(
            "brand or product criticism with no described fault to diagnose",
            "threats to switch platform; sarcasm and venting",
            "unsolicited feature requests and praise",
        ),
        excludes=(
            "angry wording around a real, diagnosable fault — the fault decides the label",
            "a refund demand — billing_and_subscription",
        ),
        examples=(
            _ex(
                "1718188__1718187",
                '[tweet-text redacted: tweet_id=1718188 sha256=4ec223075cd718f8]',
                "churn threat; fault named only vaguely",
            ),
            _ex(
                "1784923__1784922",
                '[tweet-text redacted: tweet_id=1784923 sha256=ebef936377bf0e39]',
                "boundary: comparative complaint that does invite a suggestion",
            ),
            _ex(
                "253479__253478",
                '[tweet-text redacted: tweet_id=253479 sha256=a7c1c63328dcb7dd]',
                "dissatisfaction with no specific diagnosable symptom",
            ),
        ),
        escalation_sensitive=False,
        auto_handleable=False,
        prevalence_floor=0.007,
        why_intent=(
            "An intent rather than a tone attribute: frequently no groundable resolution exists, "
            "and the support action is acknowledgement or routing to a feedback channel rather than "
            "resolution. Retrieval must not ground a reply in troubleshooting evidence when the "
            "customer is not asking for a fix. "
        ),
        confusions=("device_malfunction", "howto_information"),
    ),
    Intent(
        name="other_unclear",
        definition=(
            "A genuine AppleSupport request that fits no other label, or a message that is "
            "not a support request at all."
        ),
        includes=(
            "off-topic mentions, spam, jokes directed at the brand",
            "requests outside anything the other labels cover",
        ),
        excludes=(
            "anything a documented label covers — this is a last resort, not a shortcut",
            "messages that only lack context — set context_sufficient=false instead",
        ),
        examples=(
            _ex(
                "1330011__1330010",
                '[tweet-text redacted: tweet_id=1330011 sha256=e22515a50b10d7eb]',
                "pre-sales pricing; borderline with howto_information",
            ),
            _ex(
                "1660394__1660393",
                '[tweet-text redacted: tweet_id=1660394 sha256=a9b8c12c69b9e0bc]',
                "no identifiable problem area stated",
            ),
        ),
        escalation_sensitive=False,
        auto_handleable=False,
        prevalence_floor=0.0,
        why_intent=(
            "An outcome label on policy grounds: the message is not a support request, so no "
            "resolution evidence exists and no reply may be auto-handled. Keeping it explicit gives "
            "the classifier a target for out-of-distribution input rather than forcing a wrong "
            "intent. "
        ),
        confusions=(),
        is_catch_all=True,
    ),
)


# Tie-breaks encode the rule: the label naming the ACTION SUPPORT MUST TAKE wins over the
# label naming the customer's explanation of the cause.
_TIE_BREAKS: tuple[TieBreak, ...] = (
    TieBreak("account_access", "connectivity",
             "When a message genuinely raises both (e.g. a compromised account AND a network "
             "fault), account security wins because it is the higher-risk action. NOTE: a "
             "message about a Wi-Fi network password is NOT a tie-break case at all -- it is "
             "connectivity only, and is excluded from account_security by definition."),
    TieBreak("account_access", "billing_and_subscription",
             "A charge on a compromised account is a security incident first."),
    TieBreak("billing_and_subscription", "battery_charging",
             "When a message genuinely raises both (e.g. a disputed charge AND battery drain), "
             "money wins because it needs account-specific action. NOTE: 'charged my phone' is "
             "NOT a tie-break case -- it is battery only, and is excluded from billing by "
             "definition. Tie-breaks resolve real overlap, not lexical traps."),
    TieBreak("billing_and_subscription", "repair_order_replacement",
             "A dispute about what was billed for a repair is billing."),
    TieBreak("repair_order_replacement", "device_malfunction",
             "Physical damage or an existing repair/order makes it logistics, not diagnosis."),
    TieBreak("repair_order_replacement", "howto_information",
             "Once an order or repair exists, it is no longer a general information request."),
    TieBreak("connectivity", "device_malfunction",
             "A connectivity symptom is more specific than general malfunction."),
    TieBreak("apps_and_services", "device_malfunction",
             "One named app is more specific than the whole device."),
    TieBreak("device_malfunction", "complaint_feedback",
             "Anger about a real, diagnosable fault is still that fault."),
    TieBreak("device_malfunction", "howto_information",
             "'How do I fix this broken thing' is the fault, not an information request."),
    TieBreak("account_access", "howto_information",
             "If the customer is actually blocked from their account it is account_access; "
             "howto_information is for capability questions with no access failure."),
    TieBreak("battery_charging", "device_malfunction",
             "A battery or charging symptom is more specific than general malfunction and "
             "needs battery-specific resolution evidence."),
    TieBreak("apps_and_services", "complaint_feedback",
             "Anger about a named service that has a diagnosable fault is still that service's "
             "fault; feedback is for evaluative messages with no fault to resolve."),
    TieBreak("complaint_feedback", "howto_information",
             "An evaluative complaint with no answerable question is feedback, not a request "
             "for documentation."),
)


_ATTRIBUTES: tuple[Attribute, ...] = (
    Attribute(
        name="security_sensitive",
        definition=(
            "The customer expresses concern about compromise, unauthorised access, fraud, "
            "theft, phishing or impersonation — whether or not any of it is confirmed."
        ),
        routing_consequence=(
            "Forces ESCALATE for every intent, overriding auto-handleability. This is the "
            "hard rule in SPEC section 8, and it fires on the attribute rather than on any "
            "label."
        ),
        forces_escalation=True,
        annotation_rule=(
            "Flag on SUSPICION EXPRESSED BY THE CUSTOMER, never on confirmation: whether an "
            "account was truly compromised is not observable from the message. Measured in "
            "the train split at 0.56% of messages, of which 306 of 340 (90%) fell outside "
            "the account topic — they were stolen devices, fraudulent charges and scam apps. "
            "An intent-shaped security label would have captured a tenth of them."
        ),
    ),
    Attribute(
        name="context_sufficient",
        definition=(
            "False when the specific request cannot be determined from this message and its "
            "thread context, even where the broad topic is clear."
        ),
        routing_consequence=(
            "False forces ESCALATE or a clarifying question; a reply must never be generated "
            "for a request that has not been identified."
        ),
        forces_escalation=True,
        annotation_rule=(
            "Insufficient context is a FAILURE TO DETERMINE the request, not a kind of "
            "request. Roughly 8% of the split is bare acknowledgement ('@AppleSupport Ok'). "
            "A message may carry a clear intent AND be context-insufficient — the two are "
            "independent axes, which is why this is not an intent label."
        ),
    ),
)

_PROVENANCE = {
    "brand": "AppleSupport",
    "brand_decision": (
        "Corrected re-decision after a reply_classify defect was found. AmazonHelp and "
        "AppleSupport tied at rubric 0.7246 (exact to 6dp), broken lexicographically on "
        "pre-existing corpus features: criterion 1, absolute usable_grounding_evidence_pairs, "
        "gave AppleSupport 31,241 vs AmazonHelp 20,076 and decided it. Criteria 2-4 not "
        "reached. The original decision is preserved as superseded historical evidence; the "
        "re-decision independently reached the same brand for a different, now-authoritative "
        "reason."
    ),
    "derived_from": "AppleSupport train split only, n=60,817 English messages",
    "excluded_from_derivation": "dev, test pool, and any golden candidates — never inspected",
    "random_seed": 20260910,
    "evidence": [
        "reports/taxonomy_discovery.json",
        "reports/taxonomy_probes.json",
        "reports/taxonomy_adjudication.json",
        "reports/brand_profiles_reclassified.json",
        "reports/brand_decision_reclassified.json",
    ],
    "rejected_labels": {
        "software_update_issue": (
            "Causal attribution, not a request. battery_vs_update showed ZERO significant "
            "differences on any handling metric, so 'the update caused it' does not change "
            "what support does. A label on this basis would slice every other category in "
            "half by a criterion that changes nothing, and would be unstable for annotators."
        ),
        "account_security_compromise": (
            "Reframed as the security_sensitive attribute. 90% of security-sensitive traffic "
            "sits outside the account topic, so an intent-shaped label loses nine tenths of "
            "the safety signal."
        ),
        "privacy_data": (
            "Merged into complaint_feedback: routed to the Feedback channel in the data, the "
            "handling signature of feedback rather than of security."
        ),
        "phishing_scam_verification": (
            "Merged into howto_information: lowest deflection of the four security-adjacent "
            "groups (0.274) and answered in channel with an article. It is an information "
            "request."
        ),
        "needs_more_context": (
            "Reframed as the context_sufficient attribute: a failure to determine the "
            "request, not a kind of request."
        ),
    },
    "known_limitations": [
        "Prevalence figures are FLOORS from high-precision lexical probes, not estimates; "
        "62.9% of the split matched no probe under priority-first assignment.",
        "13 of 36 label pairs show NO material handling difference (max |diff| < 0.10). "
        "Deflection sits at 0.28-0.55 for every label because the 2017 Twitter channel was "
        "deflection-dominated, so the instrument has low discriminative power. Handling "
        "similarity is therefore treated as absence of evidence, not evidence of absence, "
        "and most boundaries rest on resolution-evidence type and escalation policy instead.",
        "The corpus is dominated by one event (the iOS 11 launch, Oct-Dec 2017); a taxonomy "
        "fitted here may not transfer to another period.",
        "security_sensitive at 0.56% yields roughly 1-2 examples in a 200-item golden set at "
        "natural prevalence. Deliberate stratified over-sampling is required, documented and "
        "reweighted at reporting time, or the safety-critical path goes untested.",
        "repair_order_replacement and account_access (floors ~1.5%) will have very wide "
        "per-class confidence intervals; point estimates must never be quoted alone.",
        "other_unclear prevalence is unmeasured — probes cannot detect 'not a support "
        "request'. Only human labelling will establish it.",
    ],
}

# Frozen 2026-09-10 after review round 3. Post-freeze changes require a NEW version and a
# decision-log entry; model performance may never motivate a change (DECISION_LOG.md D14).
TAXONOMY = Taxonomy(
    version=VERSION,
    intents=_INTENTS,
    tie_breaks=_TIE_BREAKS,
    frozen=True,
    attributes=_ATTRIBUTES,
    provenance=_PROVENANCE,
)

#: Backwards-compatible alias. The taxonomy is now frozen; the name is retained so existing
#: imports keep working rather than being silently broken by the rename.
CANDIDATE_TAXONOMY = TAXONOMY
