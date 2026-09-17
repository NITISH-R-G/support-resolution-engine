"""Dependency failures end in a typed escalation, never an unhandled crash.

Found by fault injection during the release audit. With the classifier, retriever or generator
raising anything other than an ``LLMError``, ``ReplyAgent.handle`` crashed. A crash sends no
reply, so nothing was fabricated, but it breaks the fail-closed contract: a production caller
gets an exception instead of a decision it can route to a human.

The boundaries are deliberate. Programming errors in the agent's own input validation still
raise, and ``KeyboardInterrupt`` is never swallowed. Only failures of the three external
dependencies become escalations, and the deterministic gates keep their order: a
security-sensitive message escalates for security before retrieval is ever attempted.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest

from hiver_support.agent.agent import EscalationReason, ReplyAgent
from hiver_support.agent.generation import StructuredLLMGenerator
from hiver_support.agent.llm import LLMResponseError, LLMResult
from hiver_support.agent.retrieval import EvidenceCase, RetrievalResult
from hiver_support.classifier.contract import Prediction
from hiver_support.taxonomy import TAXONOMY

WHEN = datetime(2017, 11, 1, tzinfo=timezone.utc)
EVIDENCE = (
    EvidenceCase("case0", "battery drains fast", "Check Settings > Battery > Battery Health.", WHEN, 1.0),
)
GOOD_REPLY = json.dumps({
    "response": "Check Settings > Battery > Battery Health.",
    "should_escalate": False,
    "escalation_reason": "",
    "evidence_ids": ["E1"],
    "confidence": 0.9,
})


class Classifier:
    def __init__(self, fail: BaseException | None = None, security: bool = False):
        self.fail, self.security = fail, security

    def predict(self, text):
        if self.fail is not None:
            raise self.fail
        return Prediction(
            intent="battery_charging", confidence=0.9, security_sensitive=self.security,
            context_sufficient=True, model_name="fixed", model_version="1",
            taxonomy_version=TAXONOMY.version, taxonomy_hash=TAXONOMY.frozen_hash,
        )


class Retriever:
    def __init__(self, fail: BaseException | None = None):
        self.fail, self.calls = fail, 0

    def retrieve(self, query, top_k=4, exclude_ids=None, before=None):
        self.calls += 1
        if self.fail is not None:
            raise self.fail
        return RetrievalResult(query=query, cases=EVIDENCE, confidence=0.95)


class Provider:
    name, model = "scripted", "test/model"

    def __init__(self, outcome=GOOD_REPLY):
        self.outcome, self.calls = outcome, 0

    def generate(self, prompt, *, json_mode=False):
        self.calls += 1
        if isinstance(self.outcome, BaseException):
            raise self.outcome
        return LLMResult(text=self.outcome, provider=self.name, model=self.model)

    @property
    def totals(self):
        return {}


def agent(classifier=None, retriever=None, provider=None):
    return ReplyAgent(
        classifier=classifier or Classifier(),
        retriever=retriever or Retriever(),
        generator=StructuredLLMGenerator(provider or Provider()),
    )


class TestRetrieverFailure:
    def test_a_retriever_exception_escalates_instead_of_crashing(self):
        decision = agent(retriever=Retriever(fail=RuntimeError("embedding model unavailable"))).handle(
            "my battery drains fast"
        )
        assert decision.action == "ESCALATE"
        assert decision.reason is EscalationReason.DEPENDENCY_FAILED
        assert decision.reply is None

    def test_the_failure_is_named_for_the_operator(self):
        decision = agent(retriever=Retriever(fail=RuntimeError("index missing"))).handle("battery drains")
        assert "retriever" in decision.reason_detail
        assert "RuntimeError" in decision.reason_detail

    def test_the_generator_is_never_called_after_a_retrieval_failure(self):
        provider = Provider()
        agent(retriever=Retriever(fail=RuntimeError("down")), provider=provider).handle("battery drains")
        assert provider.calls == 0


class TestClassifierFailure:
    def test_a_classifier_exception_escalates_instead_of_crashing(self):
        decision = agent(classifier=Classifier(fail=OSError("cannot load sentence-transformers"))).handle(
            "my battery drains fast"
        )
        assert decision.action == "ESCALATE"
        assert decision.reason is EscalationReason.DEPENDENCY_FAILED
        assert decision.reply is None
        assert "classifier" in decision.reason_detail

    def test_no_prediction_is_invented_for_the_failed_message(self):
        decision = agent(classifier=Classifier(fail=OSError("down"))).handle("my battery drains fast")
        assert decision.intent == "other_unclear"
        assert decision.intent_confidence == 0.0

    def test_nothing_downstream_runs_after_a_classifier_failure(self):
        retriever, provider = Retriever(), Provider()
        agent(classifier=Classifier(fail=OSError("down")), retriever=retriever, provider=provider).handle("hi there")
        assert retriever.calls == 0 and provider.calls == 0


class TestGeneratorFailure:
    def test_an_unexpected_generator_exception_escalates(self):
        decision = agent(provider=Provider(RuntimeError("boom"))).handle("my battery drains fast")
        assert decision.action == "ESCALATE"
        assert decision.reason is EscalationReason.GENERATOR_FAILED
        assert decision.reply is None

    def test_a_provider_error_still_reports_generator_failed(self):
        decision = agent(provider=Provider(LLMResponseError("401"))).handle("my battery drains fast")
        assert decision.reason is EscalationReason.GENERATOR_FAILED


class TestTheBoundariesOfWhatIsCaught:
    def test_the_security_gate_still_runs_before_retrieval_is_attempted(self):
        retriever = Retriever(fail=RuntimeError("down"))
        decision = agent(classifier=Classifier(security=True), retriever=retriever).handle("someone hacked me")
        assert decision.reason is EscalationReason.SECURITY_SENSITIVE
        assert retriever.calls == 0

    def test_invalid_input_is_still_a_programming_error(self):
        with pytest.raises(TypeError):
            agent().handle(None)

    def test_keyboard_interrupt_is_never_swallowed(self):
        with pytest.raises(KeyboardInterrupt):
            agent(retriever=Retriever(fail=KeyboardInterrupt())).handle("battery drains")

    def test_a_working_pipeline_is_unaffected(self):
        decision = agent().handle("my battery drains fast")
        assert decision.action == "AUTO_HANDLE"
        assert decision.reason is EscalationReason.NONE

    def test_dependency_failed_is_a_distinct_reason(self):
        assert EscalationReason.DEPENDENCY_FAILED.value == "dependency_failed"
        assert EscalationReason.DEPENDENCY_FAILED is not EscalationReason.GENERATOR_FAILED


class TestInjectableMasker:
    """The agent's PII masker is a parameter so the evaluated v1 configuration stays reproducible."""

    def test_the_default_masker_is_the_corrected_v2(self):
        seen = {}

        class Capture(Classifier):
            def predict(self, text):
                seen["text"] = text
                return super().predict(text)

        agent(classifier=Capture()).handle("call 1-800-555-0134 about my battery")
        assert "555" not in seen["text"]

    def test_the_v1_masker_reproduces_the_evaluated_input(self):
        from hiver_support.data.pii import mask_pii_v1

        seen = {}

        class Capture(Classifier):
            def predict(self, text):
                seen["text"] = text
                return super().predict(text)

        ReplyAgent(classifier=Capture(), retriever=Retriever(),
                   generator=StructuredLLMGenerator(Provider()), masker=mask_pii_v1,
                   ).handle("call 1-800-555-0134 about my battery")
        assert "1-800-555-0134" in seen["text"]
