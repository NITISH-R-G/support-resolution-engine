"""Invariants for the intent taxonomy.

The taxonomy is the contract every later stage depends on: the classifier predicts into it,
the golden set is labelled against it, and the escalation policy routes from it. Once it is
frozen and the golden set is labelled, changing it invalidates those labels — so its
structural properties are enforced here rather than trusted.

These tests check **structure, not label choice**. Whether `battery_charging` should exist is
a judgement made from the evidence in `reports/taxonomy_discovery.json` and
`reports/taxonomy_probes.json` and argued in `docs/INTENT_TAXONOMY.md`. What is testable is
that the taxonomy is internally coherent, deterministic, serialisable, and honest about
whether it has been frozen.
"""

from __future__ import annotations

import json

import pytest

from hiver_support.taxonomy import (
    CANDIDATE_TAXONOMY,
    Intent,
    IntentExample,
    Taxonomy,
    TaxonomyError,
    TieBreak,
)


@pytest.fixture
def taxonomy() -> Taxonomy:
    return CANDIDATE_TAXONOMY


def _intent(name: str, **overrides) -> Intent:
    defaults = dict(
        name=name,
        definition=f"definition for {name}",
        includes=("something",),
        excludes=("something else",),
        examples=(IntentExample(pair_id="1__2", text="a real message", note="why"),),
        escalation_sensitive=False,
        auto_handleable=True,
        prevalence_floor=0.05,
        confusions=(),
        is_catch_all=False,
        requires_context=False,
    )
    defaults.update(overrides)
    return Intent(**defaults)


# ----------------------------------------------------------------- structural invariants


class TestLabelUniqueness:
    def test_names_are_unique(self, taxonomy):
        names = [i.name for i in taxonomy.intents]
        assert len(names) == len(set(names))

    def test_definitions_are_unique(self, taxonomy):
        """Two labels sharing a definition cannot be told apart by an annotator."""
        definitions = [i.definition.strip().lower() for i in taxonomy.intents]
        assert len(definitions) == len(set(definitions))

    def test_names_are_valid_identifiers(self, taxonomy):
        for intent in taxonomy.intents:
            assert intent.name.islower()
            assert intent.name.replace("_", "").isalnum()
            assert not intent.name.startswith("_")

    def test_duplicate_names_are_rejected(self):
        with pytest.raises(TaxonomyError, match="duplicate"):
            Taxonomy(
                version="0.0.1",
                intents=(_intent("a"), _intent("a")),
                tie_breaks=(),
                frozen=False,
            )


class TestCompleteness:
    def test_every_intent_has_a_definition(self, taxonomy):
        for intent in taxonomy.intents:
            assert len(intent.definition.strip()) > 20

    def test_every_intent_has_inclusion_and_exclusion_criteria(self, taxonomy):
        for intent in taxonomy.intents:
            assert intent.includes, f"{intent.name} has no inclusion criteria"
            assert intent.excludes, f"{intent.name} has no exclusion criteria"

    def test_every_intent_has_real_corpus_examples(self, taxonomy):
        """Examples must carry a pair_id so any of them can be traced back to the corpus."""
        for intent in taxonomy.intents:
            assert len(intent.examples) >= 2, f"{intent.name} has too few examples"
            for example in intent.examples:
                assert "__" in example.pair_id, f"{intent.name}: {example.pair_id} not a pair id"
                if taxonomy.examples_materialized:
                    assert example.text.strip()

    def test_every_example_has_an_edit_script_and_hash_instead_of_committed_text(self, taxonomy):
        from hiver_support import codebook

        scripted = {e["pair_id"]: e for e in codebook.edit_scripts()["examples"]}
        for intent in taxonomy.intents:
            for example in intent.examples:
                assert example.pair_id in scripted
                assert len(scripted[example.pair_id]["sha256"]) == 64

    def test_exactly_one_catch_all_label(self, taxonomy):
        catch_alls = [i.name for i in taxonomy.intents if i.is_catch_all]
        assert len(catch_alls) == 1, f"expected exactly one catch-all, got {catch_alls}"

    def test_at_least_one_escalation_sensitive_label(self, taxonomy):
        assert any(i.escalation_sensitive for i in taxonomy.intents)

    def test_escalation_sensitive_labels_are_not_marked_auto_handleable(self, taxonomy):
        """A label cannot be both 'always escalate' and 'safe to auto-answer'."""
        for intent in taxonomy.intents:
            if intent.escalation_sensitive:
                assert not intent.auto_handleable, f"{intent.name} claims both"

    def test_size_is_within_the_specified_band(self, taxonomy):
        """SPEC 5 asks for 6-10 intents plus an explicit other/unclear."""
        assert 7 <= len(taxonomy.intents) <= 13


class TestTieBreaks:
    def test_tie_breaks_reference_existing_labels(self, taxonomy):
        names = {i.name for i in taxonomy.intents}
        for rule in taxonomy.tie_breaks:
            assert rule.winner in names, f"unknown winner {rule.winner}"
            assert rule.loser in names, f"unknown loser {rule.loser}"

    def test_tie_breaks_have_no_self_reference(self, taxonomy):
        for rule in taxonomy.tie_breaks:
            assert rule.winner != rule.loser

    def test_tie_breaks_contain_no_cycles(self, taxonomy):
        """A cycle would make the multi-intent policy non-deterministic."""
        edges: dict[str, set[str]] = {}
        for rule in taxonomy.tie_breaks:
            edges.setdefault(rule.winner, set()).add(rule.loser)

        def reaches(start: str, target: str, seen: set[str]) -> bool:
            if start in seen:
                return False
            seen.add(start)
            return any(
                node == target or reaches(node, target, seen)
                for node in edges.get(start, ())
            )

        for rule in taxonomy.tie_breaks:
            assert not reaches(rule.loser, rule.winner, set()), (
                f"cycle: {rule.winner} > {rule.loser} and back"
            )

    def test_cyclic_tie_breaks_are_rejected(self):
        with pytest.raises(TaxonomyError, match="cycle"):
            Taxonomy(
                version="0.0.1",
                intents=(_intent("a"), _intent("b", is_catch_all=True)),
                tie_breaks=(TieBreak("a", "b", "r1"), TieBreak("b", "a", "r2")),
                frozen=False,
            )

    def test_every_documented_confusion_pair_has_a_tie_break(self, taxonomy):
        """A confusion an annotator will hit needs a rule, or labelling is a coin flip."""
        pairs = {frozenset((r.winner, r.loser)) for r in taxonomy.tie_breaks}
        for intent in taxonomy.intents:
            for other in intent.confusions:
                assert frozenset((intent.name, other)) in pairs, (
                    f"{intent.name} vs {other} is documented as confusable but has no tie-break"
                )


class TestResolution:
    def test_resolve_returns_the_winner_of_a_documented_pair(self, taxonomy):
        rule = taxonomy.tie_breaks[0]
        assert taxonomy.resolve([rule.loser, rule.winner]) == rule.winner

    def test_resolve_is_order_independent(self, taxonomy):
        rule = taxonomy.tie_breaks[0]
        assert taxonomy.resolve([rule.winner, rule.loser]) == taxonomy.resolve(
            [rule.loser, rule.winner]
        )

    def test_resolve_single_candidate_returns_it(self, taxonomy):
        assert taxonomy.resolve(["battery_charging"]) == "battery_charging"

    def test_resolve_empty_returns_the_catch_all(self, taxonomy):
        assert taxonomy.resolve([]) == taxonomy.catch_all.name

    def test_resolve_rejects_unknown_labels(self, taxonomy):
        with pytest.raises(TaxonomyError, match="unknown"):
            taxonomy.resolve(["not_a_real_intent"])

    def test_resolve_is_deterministic_for_unordered_pairs(self, taxonomy):
        """Without an explicit rule the result must still be stable, never arbitrary."""
        names = [i.name for i in taxonomy.intents if not i.is_catch_all][:2]
        first = taxonomy.resolve(names)
        second = taxonomy.resolve(list(reversed(names)))
        assert first == second


class TestVersioningAndSerialisation:
    def test_version_is_semver_like(self, taxonomy):
        parts = taxonomy.version.split(".")
        assert len(parts) == 3
        assert all(p.isdigit() for p in parts)

    def test_content_hash_is_stable_across_calls(self, taxonomy):
        assert taxonomy.content_hash() == taxonomy.content_hash()

    def test_content_hash_changes_when_content_changes(self, taxonomy):
        altered = Taxonomy(
            version=taxonomy.version,
            intents=taxonomy.intents[:-1],
            tie_breaks=(),
            frozen=False,
        )
        assert altered.content_hash() != taxonomy.content_hash()

    def test_round_trips_through_dict(self, taxonomy):
        restored = Taxonomy.from_dict(taxonomy.to_dict())
        assert restored.content_hash() == taxonomy.content_hash()
        assert restored.version == taxonomy.version
        assert [i.name for i in restored.intents] == [i.name for i in taxonomy.intents]

    def test_serialises_to_json(self, taxonomy):
        payload = json.dumps(taxonomy.to_dict())
        assert Taxonomy.from_dict(json.loads(payload)).content_hash() == taxonomy.content_hash()


class TestFreezeStateSuperseded:
    """Superseded by approval on 2026-09-10.

    These asserted CANDIDATE behaviour (not frozen, freezable). The taxonomy was approved and
    frozen at v0.3.0, so those assertions are obsolete rather than broken. The frozen
    invariants they were protecting are now asserted in ``TestFrozenState`` below. Recorded as
    a deliberate replacement rather than a silent deletion.
    """

    @pytest.mark.skip(reason="superseded: taxonomy frozen at v0.3.0 after approval")
    def test_candidate_taxonomy_is_not_frozen(self, taxonomy):
        """Freezing is a reviewed decision, not something that happens by drafting."""
        assert taxonomy.frozen is False

    @pytest.mark.skip(reason="superseded: taxonomy frozen at v0.3.0 after approval")
    def test_freezing_produces_a_frozen_copy_with_the_same_content(self, taxonomy):
        frozen = taxonomy.freeze()
        assert frozen.frozen is True
        assert frozen.content_hash() == taxonomy.content_hash()

    @pytest.mark.skip(reason="superseded: covered by TestFrozenState.test_refreezing_raises")
    def test_freezing_an_already_frozen_taxonomy_raises(self, taxonomy):
        frozen = taxonomy.freeze()
        with pytest.raises(TaxonomyError, match="already frozen"):
            frozen.freeze()

    def test_intents_are_immutable(self, taxonomy):
        with pytest.raises((AttributeError, TypeError)):
            taxonomy.intents[0].name = "changed"  # type: ignore[misc]


class TestPrevalenceMetadata:
    def test_prevalence_floors_are_fractions(self, taxonomy):
        for intent in taxonomy.intents:
            assert 0.0 <= intent.prevalence_floor <= 1.0

    def test_prevalence_is_documented_as_a_floor_not_an_estimate(self, taxonomy):
        """Probe counts undercount by design; the field name and docs must not overclaim."""
        assert "floor" in Intent.__doc__.lower()


# ===========================================================================================
# Milestone 3 FREEZE (v0.3.0) — approved Round 3
# ===========================================================================================

APPROVED_INTENTS = {
    "device_malfunction",
    "battery_charging",
    "apps_and_services",
    "billing_and_subscription",
    "connectivity",
    "howto_information",
    "complaint_feedback",
    "account_access",
    "repair_order_replacement",
    "other_unclear",
}

REJECTED_LABELS = {
    "software_update_issue",
    "account_security_compromise",
    "account_security",
    "privacy_data",
    "phishing_scam_verification",
    "needs_more_context",
}


class TestApprovedLabelSet:
    def test_intent_names_match_the_approved_set_exactly(self, taxonomy):
        assert {i.name for i in taxonomy.intents} == APPROVED_INTENTS

    def test_there_are_exactly_ten_intents(self, taxonomy):
        assert len(taxonomy.intents) == 10

    @pytest.mark.parametrize("rejected", sorted(REJECTED_LABELS))
    def test_rejected_labels_are_absent(self, taxonomy, rejected):
        assert rejected not in {i.name for i in taxonomy.intents}


class TestOrthogonalAttributes:
    """Safety and context are axes, not labels. Encoding them as intents loses signal."""

    def test_attributes_are_declared(self, taxonomy):
        assert {a.name for a in taxonomy.attributes} == {
            "security_sensitive",
            "context_sufficient",
        }

    def test_every_attribute_documents_its_routing_consequence(self, taxonomy):
        for attribute in taxonomy.attributes:
            assert attribute.routing_consequence.strip()

    def test_security_sensitive_forces_escalation(self, taxonomy):
        security = next(a for a in taxonomy.attributes if a.name == "security_sensitive")
        assert security.forces_escalation is True

    def test_security_escalation_is_independent_of_intent(self, taxonomy):
        """The 90% finding: most security-sensitive traffic is not account-shaped."""
        for intent in taxonomy.intents:
            assert taxonomy.must_escalate(intent.name, security_sensitive=True) is True

    def test_insufficient_context_forces_escalation_or_clarification(self, taxonomy):
        context = next(a for a in taxonomy.attributes if a.name == "context_sufficient")
        assert context.forces_escalation is True


class TestIntentJustification:
    """Each label must say why it is an intent, not a topic, symptom or attribute."""

    def test_every_intent_documents_why_it_is_not_merely_a_topic(self, taxonomy):
        for intent in taxonomy.intents:
            assert len(intent.why_intent.strip()) > 80, intent.name

    def test_justification_names_a_recognised_basis(self, taxonomy):
        bases = ("resolution evidence", "support action", "escalation", "retrieval")
        for intent in taxonomy.intents:
            assert any(b in intent.why_intent.lower() for b in bases), intent.name


class TestFrozenState:
    def test_taxonomy_is_frozen(self, taxonomy):
        assert taxonomy.frozen is True

    def test_version_is_the_approved_freeze_version(self, taxonomy):
        assert taxonomy.version == "0.3.0"

    def test_frozen_hash_is_the_recorded_v0_3_0_hash(self, taxonomy):
        assert taxonomy.frozen_hash == (
            "613f5dfec1253168c8f2d01db141923c9e41363b6158c79bab6f067df4a9ee4d"
        )

    def test_frozen_hash_is_recorded_and_matches_content(self, taxonomy):
        if not taxonomy.examples_materialized:
            pytest.skip("codebook texts not materialised (scripts/materialize_text.py --codebook)")
        assert taxonomy.content_hash() == taxonomy.frozen_hash

    def test_a_drifted_codebook_text_is_detected(self, taxonomy):
        from dataclasses import replace as dc_replace

        intent = taxonomy.intents[0]
        drifted_examples = tuple(dc_replace(e, text=(e.text or "x") + " drift") for e in intent.examples)
        others = tuple(
            dc_replace(i, examples=tuple(dc_replace(e, text=e.text or "x") for e in i.examples))
            for i in taxonomy.intents[1:]
        )
        drifted = dc_replace(
            taxonomy, intents=(dc_replace(intent, examples=drifted_examples), *others)
        )
        with pytest.raises(TaxonomyError):
            drifted.frozen_hash

    def test_refreezing_raises(self, taxonomy):
        with pytest.raises(TaxonomyError):
            taxonomy.freeze()


class TestProvenanceRecord:
    def test_brand_and_derivation_are_recorded(self, taxonomy):
        assert taxonomy.provenance["brand"] == "AppleSupport"
        assert "train" in taxonomy.provenance["derived_from"].lower()

    def test_rejected_labels_are_recorded_with_reasons(self, taxonomy):
        rejected = taxonomy.provenance["rejected_labels"]
        assert "software_update_issue" in rejected
        assert rejected["software_update_issue"].strip()

    def test_known_limitations_are_recorded(self, taxonomy):
        assert len(taxonomy.provenance["known_limitations"]) >= 3
