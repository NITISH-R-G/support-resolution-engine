"""The LLM generation contract and its fail-closed wiring into the agent.

The model is treated as untrusted throughout. It is given evidence and told what it may not
do, but the prompt is a *request*, not a guarantee — so everything safety-relevant is checked
afterwards by code the model cannot influence:

* Its ``should_escalate`` may only **add** escalation. A model that could clear a deterministic
  gate would be setting its own safety policy, and an LLM that decides when to escalate is
  unauditable.
* Its ``evidence_ids`` must be a subset of what retrieval actually returned. A cited case that
  was never retrieved is a fabricated citation, which is worse than no citation because it
  looks verifiable.
* A malformed response escalates. It never becomes an empty reply, because an empty reply is
  indistinguishable from the model declining and would convert an outage into a routing change.
* The independent grounding validator still runs on whatever survives.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import pytest

from hiver_support.agent.agent import EscalationReason, ReplyAgent
from hiver_support.agent.generation import (
    STRUCTURED_PROMPT_VERSION,
    EvidenceTemplateGenerator,
    GeneratedReply,
    StructuredLLMGenerator,
)
from hiver_support.agent.llm import LLMNotConfiguredError, LLMResponseError, LLMResult, NullProvider
from hiver_support.agent.retrieval import EvidenceCase

WHEN = datetime(2017, 11, 1, tzinfo=timezone.utc)


def evidence(n: int = 2) -> tuple[EvidenceCase, ...]:
    return tuple(
        EvidenceCase(
            case_id=f"case{i}",
            customer_text=f"my battery drains quickly {i}",
            resolution_text="Check Settings > Battery > Battery Health and restart the device.",
            created_at=WHEN - timedelta(days=i + 1),
            score=1.0 - i * 0.1,
        )
        for i in range(n)
    )


class ScriptedProvider:
    """Returns whatever the test tells it to, including garbage."""

    name = "scripted"
    model = "test/model"

    def __init__(self, *payloads):
        self.payloads = list(payloads)
        self.prompts: list[str] = []

    def generate(self, prompt: str, *, json_mode: bool = False) -> LLMResult:
        self.prompts.append(prompt)
        outcome = self.payloads.pop(0) if self.payloads else "{}"
        if isinstance(outcome, Exception):
            raise outcome
        text = outcome if isinstance(outcome, str) else json.dumps(outcome)
        return LLMResult(
            text=text, provider=self.name, model=self.model,
            prompt_tokens=100, completion_tokens=40, cost_usd=0.0001, latency_ms=250.0,
        )

    @property
    def totals(self) -> dict:
        return {"requests": len(self.prompts), "failures": 0, "cost_usd": 0.0}


def a_reply(**overrides) -> dict:
    payload = {
        "response": "Please check Settings > Battery > Battery Health, then restart.",
        "should_escalate": False,
        "escalation_reason": "",
        "evidence_ids": ["E1"],
        "confidence": 0.8,
    }
    payload.update(overrides)
    return payload


class TestThePromptGivesTheModelWhatTheContractRequires:
    def test_it_contains_the_customer_message_intent_evidence_and_risk_state(self):
        provider = ScriptedProvider(a_reply())
        StructuredLLMGenerator(provider).generate(
            "my battery dies in an hour",
            evidence(),
            "battery_charging",
            security_sensitive=False,
            context_sufficient=True,
        )
        prompt = provider.prompts[0]
        assert "my battery dies in an hour" in prompt
        assert "battery_charging" in prompt
        assert "[E1]" in prompt
        assert "Battery Health" in prompt
        assert "security" in prompt.lower()

    def test_it_states_the_absolute_constraints(self):
        provider = ScriptedProvider(a_reply())
        StructuredLLMGenerator(provider).generate("hi", evidence(), "battery_charging")
        prompt = provider.prompts[0].lower()
        for forbidden in ("refund", "polic", "never claim", "insufficient_evidence"):
            assert forbidden in prompt

    def test_the_prompt_is_versioned_so_a_cache_entry_cannot_outlive_its_instructions(self):
        assert STRUCTURED_PROMPT_VERSION
        assert StructuredLLMGenerator(ScriptedProvider()).prompt_version == (
            STRUCTURED_PROMPT_VERSION
        )

    def test_json_mode_is_requested(self):
        class Recorder(ScriptedProvider):
            def __init__(self):
                super().__init__(a_reply())
                self.json_modes = []

            def generate(self, prompt, *, json_mode=False):
                self.json_modes.append(json_mode)
                return super().generate(prompt, json_mode=json_mode)

        recorder = Recorder()
        StructuredLLMGenerator(recorder).generate("hi", evidence(), "battery_charging")
        assert recorder.json_modes == [True]


class TestStructuredOutputParsing:
    def test_a_valid_structured_reply_is_parsed(self):
        draft = StructuredLLMGenerator(ScriptedProvider(a_reply())).generate(
            "hi", evidence(), "battery_charging"
        )
        assert isinstance(draft, GeneratedReply)
        assert draft.text.startswith("Please check")
        assert draft.cited_case_ids == ("case0",)
        assert draft.model_should_escalate is False
        assert draft.model_confidence == pytest.approx(0.8)

    def test_json_wrapped_in_markdown_fences_is_still_parsed(self):
        fenced = "```json\n" + json.dumps(a_reply()) + "\n```"
        draft = StructuredLLMGenerator(ScriptedProvider(fenced)).generate(
            "hi", evidence(), "battery_charging"
        )
        assert draft.text.startswith("Please check")

    def test_usage_and_provenance_travel_with_the_draft(self):
        draft = StructuredLLMGenerator(ScriptedProvider(a_reply())).generate(
            "hi", evidence(), "battery_charging"
        )
        assert draft.usage["prompt_tokens"] == 100
        assert draft.usage["model"] == "test/model"
        assert "test/model" in draft.generator_name

    @pytest.mark.parametrize(
        "bad",
        [
            "not json at all",
            "{",
            json.dumps({"should_escalate": False}),
            json.dumps({"response": "hi"}),
            json.dumps({"response": "hi", "should_escalate": "maybe"}),
            json.dumps({"response": 42, "should_escalate": False}),
        ],
    )
    def test_a_response_violating_the_schema_raises(self, bad):
        with pytest.raises(LLMResponseError):
            StructuredLLMGenerator(ScriptedProvider(bad)).generate(
                "hi", evidence(), "battery_charging"
            )

    def test_an_out_of_range_confidence_raises_rather_than_being_clamped(self):
        # Clamping would turn a model that does not understand the scale into one that looks
        # calibrated.
        with pytest.raises(LLMResponseError, match="confidence"):
            StructuredLLMGenerator(ScriptedProvider(a_reply(confidence=4.2))).generate(
                "hi", evidence(), "battery_charging"
            )

    def test_a_missing_confidence_is_allowed_and_recorded_as_unknown(self):
        payload = a_reply()
        payload.pop("confidence")
        draft = StructuredLLMGenerator(ScriptedProvider(payload)).generate(
            "hi", evidence(), "battery_charging"
        )
        assert draft.model_confidence is None


class TestFabricatedCitationsAreRejected:
    def test_an_evidence_label_that_was_never_offered_raises(self):
        # Superseded contract: the model cites opaque labels, so an unknown label - not an
        # unknown id - is what fabrication looks like now.
        with pytest.raises(LLMResponseError, match="not a known evidence label"):
            StructuredLLMGenerator(
                ScriptedProvider(a_reply(evidence_ids=["E1", "E99"]))
            ).generate("hi", evidence(), "battery_charging")

    def test_citing_a_subset_of_retrieved_evidence_is_fine(self):
        draft = StructuredLLMGenerator(
            ScriptedProvider(a_reply(evidence_ids=["E2"]))
        ).generate("hi", evidence(), "battery_charging")
        assert draft.cited_case_ids == ("case1",)

    def test_citing_nothing_while_escalating_is_fine(self):
        draft = StructuredLLMGenerator(
            ScriptedProvider(a_reply(should_escalate=True, evidence_ids=[], response=""))
        ).generate("hi", evidence(), "battery_charging")
        assert draft.model_should_escalate is True


class TestTheModelsOwnEscapeHatches:
    def test_the_insufficient_evidence_sentinel_becomes_an_empty_draft(self):
        draft = StructuredLLMGenerator(
            ScriptedProvider(a_reply(response="INSUFFICIENT_EVIDENCE"))
        ).generate("hi", evidence(), "battery_charging")
        assert draft.text == ""

    def test_no_evidence_means_no_call_is_made_at_all(self):
        provider = ScriptedProvider(a_reply())
        draft = StructuredLLMGenerator(provider).generate("hi", (), "battery_charging")
        assert draft.text == ""
        assert provider.prompts == []

    def test_an_unconfigured_provider_still_refuses_loudly(self):
        with pytest.raises(LLMNotConfiguredError):
            StructuredLLMGenerator(NullProvider()).generate("hi", evidence(), "battery_charging")


class TestTheDeterministicGeneratorIsUnaffected:
    def test_it_still_works_with_no_provider_and_no_key(self):
        draft = EvidenceTemplateGenerator().generate("hi", evidence(), "battery_charging")
        assert "Battery Health" in draft.text
        assert draft.cited_case_ids == ("case0",)

    def test_it_reports_no_model_opinion_at_all(self):
        # The template generator has no view on escalation; the deterministic gates decide.
        draft = EvidenceTemplateGenerator().generate("hi", evidence(), "battery_charging")
        assert draft.model_should_escalate is None
        assert draft.model_confidence is None

    def test_it_accepts_the_risk_arguments_without_using_them(self):
        draft = EvidenceTemplateGenerator().generate(
            "hi", evidence(), "battery_charging", security_sensitive=True, context_sufficient=False
        )
        assert draft.text


class FixedClassifier:
    def __init__(self, intent="battery_charging", security=False, context=True):
        from hiver_support.classifier.contract import Prediction
        from hiver_support.taxonomy import TAXONOMY

        self._prediction = Prediction(
            intent=intent,
            confidence=0.9,
            security_sensitive=security,
            context_sufficient=context,
            model_name="fixed",
            model_version="1.0.0",
            taxonomy_version=TAXONOMY.version,
            taxonomy_hash=TAXONOMY.frozen_hash,
        )

    def predict(self, _text):
        return self._prediction


class FixedRetriever:
    def __init__(self, cases=None, confidence=0.95):
        from hiver_support.agent.retrieval import RetrievalResult

        self._result = RetrievalResult(
            query="q", cases=cases if cases is not None else evidence(), confidence=confidence
        )

    def retrieve(self, query, top_k=4, exclude_ids=None, before=None):
        return self._result


class TestTheAgentTreatsTheModelAsUntrusted:
    def test_a_model_escalation_is_honoured(self):
        agent = ReplyAgent(
            classifier=FixedClassifier(),
            retriever=FixedRetriever(),
            generator=StructuredLLMGenerator(
                ScriptedProvider(
                    a_reply(
                        should_escalate=True,
                        escalation_reason="evidence does not address this",
                        response="I am not sure.",
                    )
                )
            ),
        )
        decision = agent.handle("my battery dies in an hour")
        assert decision.action == "ESCALATE"
        assert decision.reason is EscalationReason.MODEL_REQUESTED
        assert "evidence does not address this" in decision.reason_detail

    def test_a_model_cannot_clear_the_security_gate(self):
        # The deterministic gate runs first and the model is never consulted, so a model that
        # wanted to auto-handle a security case could not express it.
        provider = ScriptedProvider(a_reply(should_escalate=False))
        agent = ReplyAgent(
            classifier=FixedClassifier(security=True),
            retriever=FixedRetriever(),
            generator=StructuredLLMGenerator(provider),
        )
        decision = agent.handle("someone has hacked my account")
        assert decision.action == "ESCALATE"
        assert decision.reason is EscalationReason.SECURITY_SENSITIVE
        assert provider.prompts == []

    def test_a_model_cannot_clear_the_context_gate(self):
        agent = ReplyAgent(
            classifier=FixedClassifier(context=False),
            retriever=FixedRetriever(),
            generator=StructuredLLMGenerator(ScriptedProvider(a_reply(should_escalate=False))),
        )
        assert agent.handle("ok").reason is EscalationReason.INSUFFICIENT_CONTEXT

    def test_a_model_cannot_clear_the_policy_gate(self):
        agent = ReplyAgent(
            classifier=FixedClassifier(intent="billing_and_subscription"),
            retriever=FixedRetriever(),
            generator=StructuredLLMGenerator(ScriptedProvider(a_reply(should_escalate=False))),
        )
        assert agent.handle("charge me twice again").reason is EscalationReason.POLICY_INTENT

    def test_a_malformed_model_response_escalates_rather_than_crashing_the_run(self):
        agent = ReplyAgent(
            classifier=FixedClassifier(),
            retriever=FixedRetriever(),
            generator=StructuredLLMGenerator(ScriptedProvider("not json")),
        )
        decision = agent.handle("my battery dies in an hour")
        assert decision.action == "ESCALATE"
        assert decision.reason is EscalationReason.GENERATOR_FAILED

    def test_a_provider_outage_escalates_rather_than_producing_an_empty_reply(self):
        agent = ReplyAgent(
            classifier=FixedClassifier(),
            retriever=FixedRetriever(),
            generator=StructuredLLMGenerator(ScriptedProvider(LLMResponseError("503"))),
        )
        decision = agent.handle("my battery dies in an hour")
        assert decision.action == "ESCALATE"
        assert decision.reason is EscalationReason.GENERATOR_FAILED
        assert decision.reply is None

    def test_grounding_still_runs_on_a_model_reply_that_claimed_an_action(self):
        agent = ReplyAgent(
            classifier=FixedClassifier(),
            retriever=FixedRetriever(),
            generator=StructuredLLMGenerator(
                ScriptedProvider(a_reply(response="I have issued a refund to your account."))
            ),
        )
        decision = agent.handle("my battery dies in an hour")
        assert decision.action == "ESCALATE"
        assert decision.reason is EscalationReason.UNGROUNDED

    def test_a_clean_model_reply_can_be_auto_handled(self):
        agent = ReplyAgent(
            classifier=FixedClassifier(),
            retriever=FixedRetriever(),
            generator=StructuredLLMGenerator(ScriptedProvider(a_reply())),
        )
        decision = agent.handle("my battery dies in an hour")
        assert decision.action == "AUTO_HANDLE"
        assert decision.evidence_ids == ("case0",)

    def test_the_decision_records_which_model_produced_the_reply(self):
        agent = ReplyAgent(
            classifier=FixedClassifier(),
            retriever=FixedRetriever(),
            generator=StructuredLLMGenerator(ScriptedProvider(a_reply())),
        )
        payload = agent.handle("my battery dies in an hour").to_dict()
        assert "test/model" in payload["generator"]
        assert payload["usage"]["prompt_tokens"] == 100


class TestOpaqueEvidenceLabels:
    """The model cites E1/E2; canonical ids never leave this process.

    Milestone 6 measured the cost of the old contract. Real models could not reproduce
    compound ids like ``387511__387510``: Groq echoed the ``case `` label we printed, and
    OpenRouter truncated at the ``__`` and returned ``387511``. Both were rejected as
    fabricated citations, so a third of one run measured our id format rather than model
    behaviour.

    Opaque labels remove the failure mode instead of tolerating it. There is no prefix match
    and no fuzzy match anywhere below: a label either resolves exactly or the response is
    refused. The mapping happens outside the model, so a model that mangles a label cannot
    bind a citation to the wrong case — it can only fail closed.
    """

    @staticmethod
    def _labels(cases):
        from hiver_support.agent.generation import build_evidence_labels

        return build_evidence_labels(cases)

    def _parse(self, cited, cases=None):
        from hiver_support.agent.generation import _parse_structured

        cases = cases if cases is not None else evidence(3)
        return _parse_structured(
            json.dumps(
                {
                    "response": "hi",
                    "should_escalate": False,
                    "evidence_ids": cited,
                    "confidence": 0.5,
                }
            ),
            self._labels(cases),
        )

    # ---------------------------------------------------------------- mapping construction

    def test_labels_are_sequential_opaque_and_one_based(self):
        assert list(self._labels(evidence(3))) == ["E1", "E2", "E3"]

    def test_a_label_maps_to_the_case_in_the_same_position(self):
        cases = evidence(3)
        labels = self._labels(cases)
        assert labels["E1"] == cases[0].case_id
        assert labels["E2"] == cases[1].case_id
        assert labels["E3"] == cases[2].case_id

    def test_no_label_maps_to_the_wrong_canonical_id(self):
        # The failure this guards is silent and permanent: a reply attributed to evidence it
        # was not built from cannot be audited afterwards.
        cases = evidence(3)
        labels = self._labels(cases)
        for index, (label, case_id) in enumerate(labels.items()):
            assert case_id == cases[index].case_id
            assert sum(1 for v in labels.values() if v == case_id) == 1

    def test_an_empty_evidence_set_produces_no_labels(self):
        assert self._labels(()) == {}

    # ---------------------------------------------------------------- the model never sees ids

    def test_the_prompt_shows_labels_and_never_a_canonical_id(self):
        provider = ScriptedProvider(a_reply(evidence_ids=["E1"]))
        cases = evidence(2)
        StructuredLLMGenerator(provider).generate("hi", cases, "battery_charging")
        prompt = provider.prompts[0]
        assert "[E1]" in prompt and "[E2]" in prompt
        for case in cases:
            assert case.case_id not in prompt, "a raw database id reached the model"

    def test_the_prompt_asks_for_labels_only(self):
        provider = ScriptedProvider(a_reply(evidence_ids=["E1"]))
        StructuredLLMGenerator(provider).generate("hi", evidence(2), "battery_charging")
        assert "E1" in provider.prompts[0]

    # ---------------------------------------------------------------- the eight required cases

    def test_1_a_correct_single_label_resolves_to_its_canonical_id(self):
        cases = evidence(3)
        assert self._parse(["E1"], cases)["evidence_ids"] == [cases[0].case_id]

    def test_2_multiple_labels_resolve_in_the_order_cited(self):
        cases = evidence(3)
        assert self._parse(["E3", "E1"], cases)["evidence_ids"] == [
            cases[2].case_id,
            cases[0].case_id,
        ]

    def test_3_an_unknown_label_fails_closed(self):
        with pytest.raises(LLMResponseError, match="E99"):
            self._parse(["E99"])

    def test_4_a_raw_canonical_id_fails_closed(self):
        # The model was never shown this id, so producing one is fabrication, not obedience.
        cases = evidence(3)
        with pytest.raises(LLMResponseError, match="not a known evidence label"):
            self._parse([cases[0].case_id], cases)

    def test_5_a_truncated_canonical_id_fails_closed(self):
        # OpenRouter returned "387511" for "387511__387510". No prefix matching: accepting it
        # could bind the citation to a different case entirely.
        with pytest.raises(LLMResponseError, match="not a known evidence label"):
            self._parse(["case0_truncated"])

    def test_6_duplicate_labels_are_deduplicated_deterministically(self):
        cases = evidence(3)
        assert self._parse(["E1", "E1", "E2", "E1"], cases)["evidence_ids"] == [
            cases[0].case_id,
            cases[1].case_id,
        ]

    def test_7_citing_nothing_is_allowed(self):
        assert self._parse([])["evidence_ids"] == []

    def test_8_a_label_outside_the_supplied_range_fails_closed(self):
        # Two cases were supplied, so E3 does not exist for this request even though it is a
        # well-formed label in general.
        with pytest.raises(LLMResponseError, match="E3"):
            self._parse(["E3"], evidence(2))

    # ---------------------------------------------------------------- formatting tolerance

    @pytest.mark.parametrize("written", ["E1", "e1", "[E1]", " E1 ", "[e1]"])
    def test_label_formatting_is_normalised_but_matching_stays_exact(self, written):
        # Normalising case and brackets cannot mis-bind: the label set is closed and lookup is
        # exact afterwards. This is not fuzzy matching.
        cases = evidence(3)
        assert self._parse([written], cases)["evidence_ids"] == [cases[0].case_id]

    def test_a_label_is_never_matched_by_prefix(self):
        cases = evidence(12)
        labels = self._labels(cases)
        assert "E1" in labels and "E12" in labels
        assert self._parse(["E12"], cases)["evidence_ids"] == [cases[11].case_id]

    def test_a_non_string_citation_is_refused(self):
        with pytest.raises(LLMResponseError, match="list of strings"):
            self._parse([1])

    # ---------------------------------------------------------------- end to end

    def test_the_agent_auto_handles_a_label_citation_and_records_canonical_ids(self):
        agent = ReplyAgent(
            classifier=FixedClassifier(),
            retriever=FixedRetriever(),
            generator=StructuredLLMGenerator(
                ScriptedProvider(a_reply(evidence_ids=["E1"]))
            ),
        )
        decision = agent.handle("my battery dies in an hour")
        assert decision.action == "AUTO_HANDLE"
        assert decision.evidence_ids == ("case0",)

    def test_the_agent_escalates_when_the_model_invents_a_label(self):
        agent = ReplyAgent(
            classifier=FixedClassifier(),
            retriever=FixedRetriever(),
            generator=StructuredLLMGenerator(
                ScriptedProvider(a_reply(evidence_ids=["E7"]))
            ),
        )
        decision = agent.handle("my battery dies in an hour")
        assert decision.action == "ESCALATE"
        assert decision.reason is EscalationReason.GENERATOR_FAILED

    def test_only_canonical_ids_are_persisted(self):
        agent = ReplyAgent(
            classifier=FixedClassifier(),
            retriever=FixedRetriever(),
            generator=StructuredLLMGenerator(
                ScriptedProvider(a_reply(evidence_ids=["E1", "E2"]))
            ),
        )
        payload = agent.handle("my battery dies in an hour").to_dict()
        assert payload["evidence_ids"] == ["case0", "case1"]
        for value in payload["evidence_ids"]:
            assert not value.startswith("E"), "an opaque label was persisted as evidence"

    def test_the_prompt_version_records_the_new_contract(self):
        # The cache key includes the prompt version, so entries written under the old
        # compound-id contract cannot be served for the new one.
        assert STRUCTURED_PROMPT_VERSION != "structured-v1"
