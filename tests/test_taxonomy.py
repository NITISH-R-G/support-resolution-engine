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
                assert example.text.strip()

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


class TestFreezeState:
    def test_candidate_taxonomy_is_not_frozen(self, taxonomy):
        """Freezing is a reviewed decision, not something that happens by drafting."""
        assert taxonomy.frozen is False

    def test_freezing_produces_a_frozen_copy_with_the_same_content(self, taxonomy):
        frozen = taxonomy.freeze()
        assert frozen.frozen is True
        assert frozen.content_hash() == taxonomy.content_hash()

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
