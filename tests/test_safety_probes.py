"""The diagnostic probe set: complete, synthetic, and never treated as evaluation data.

The probes are the one place in this project where inputs are written rather than sampled. So
the tests here guard the two things that could quietly go wrong with that: the set silently
losing coverage of a failure mode, and a synthetic message drifting into a place where it
could be mistaken for corpus or evaluation data.
"""

from __future__ import annotations

import pytest

from hiver_support.agent.probes import SAFETY_PROBES, ProbeKind, SafetyProbe, probes_by_kind


class TestCoverage:
    def test_every_failure_mode_has_at_least_one_probe(self):
        covered = {probe.kind for probe in SAFETY_PROBES}
        missing = sorted(k.value for k in ProbeKind if k not in covered)
        assert missing == [], f"failure modes with no probe: {missing}"

    def test_the_eleven_categories_the_milestone_asked_for_are_present(self):
        kinds = probes_by_kind()
        for required in (
            "ambiguous",
            "multi_intent",
            "security_sensitive",
            "insufficient_context",
            "irrelevant_evidence",
            "outdated_evidence",
            "deflecting_evidence",
            "unsupported_action",
            "paraphrase",
            "unhelpful_but_groundable",
        ):
            assert required in kinds

    def test_prompt_injection_is_probed_too(self):
        assert "prompt_injection" in probes_by_kind()

    def test_probe_ids_are_unique(self):
        ids = [probe.probe_id for probe in SAFETY_PROBES]
        assert len(set(ids)) == len(ids)


class TestEveryProbeIsUsable:
    @pytest.mark.parametrize("probe", SAFETY_PROBES, ids=lambda p: p.probe_id)
    def test_a_probe_carries_a_message_and_what_to_look_for(self, probe: SafetyProbe):
        assert probe.message.strip()
        assert len(probe.what_to_watch_for) > 40, (
            "a probe without a stated expectation is a message, not a diagnostic"
        )

    @pytest.mark.parametrize("probe", SAFETY_PROBES, ids=lambda p: p.probe_id)
    def test_a_probe_is_always_marked_synthetic(self, probe: SafetyProbe):
        assert probe.is_synthetic is True
        assert probe.to_dict()["is_synthetic"] is True

    @pytest.mark.parametrize("probe", SAFETY_PROBES, ids=lambda p: p.probe_id)
    def test_serialised_probes_declare_their_provenance(self, probe: SafetyProbe):
        provenance = probe.to_dict()["provenance"]
        assert "SYNTHETIC" in provenance
        assert "never evaluation data" in provenance


class TestProbesNeverBecomeEvaluationData:
    def test_no_probe_text_appears_in_the_golden_candidate_set(self):
        # A synthetic message in the golden set would be a fabricated evaluation example.
        from pathlib import Path

        candidates = Path(__file__).resolve().parents[1] / "data" / "golden" / "candidates.jsonl"
        if not candidates.exists():
            pytest.skip("golden candidate set not built on this machine")
        raw = candidates.read_text(encoding="utf-8")
        for probe in SAFETY_PROBES:
            assert probe.message[:60] not in raw

    def test_probes_are_not_labelled_and_have_no_label_field(self):
        for probe in SAFETY_PROBES:
            keys = probe.to_dict()
            assert "intent" not in keys
            assert "should_escalate" not in keys
