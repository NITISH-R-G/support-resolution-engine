"""Routing invariants that nothing downstream may weaken.

Milestone 6 measured the failure these pin. A described Apple ID takeover was AUTO_HANDLED,
and the model that saw it reported ``should_escalate: false`` with confidence 0.95. Two layers
disagreed with reality at once, so the guarantee cannot live in either of them.

The invariant is therefore enforced at routing, exhaustively rather than by example: for
**every** intent in the frozen taxonomy and across the full confidence range, a
security-sensitive message escalates. No retrieval score, model confidence, reply quality,
intent or provider can reach that decision.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from hiver_support.agent.agent import EscalationReason, ReplyAgent
from hiver_support.agent.generation import StructuredLLMGenerator
from hiver_support.agent.llm import LLMResult
from hiver_support.agent.retrieval import EvidenceCase, RetrievalResult
from hiver_support.classifier.contract import Prediction
from hiver_support.taxonomy import TAXONOMY

WHEN = datetime(2017, 11, 1, tzinfo=timezone.utc)
CONFIDENCES = [0.0, 0.25, 0.5, 0.75, 0.99, 1.0]


def evidence(n: int = 2) -> tuple[EvidenceCase, ...]:
    return tuple(
        EvidenceCase(
            case_id=f"case{i}",
            customer_text="my device is misbehaving",
            resolution_text="Please restart the device and let us know.",
            created_at=WHEN - timedelta(days=i + 1),
            score=1.0 - i * 0.1,
        )
        for i in range(n)
    )


class FixedClassifier:
    """Emits exactly the prediction a test asks for, including impossible combinations."""

    def __init__(self, intent, confidence, security, context=True):
        self._prediction = Prediction(
            intent=intent,
            confidence=confidence,
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
    def __init__(self, confidence=0.99):
        self._result = RetrievalResult(query="q", cases=evidence(), confidence=confidence)

    def retrieve(self, query, top_k=4, exclude_ids=None, before=None):
        return self._result


class EagerProvider:
    """A model doing its confident best to get the reply sent.

    Deliberately adversarial: it insists on not escalating, at maximum confidence, with a
    clean grounded reply. If routing can be talked out of a security escalation, this is what
    would do it.
    """

    name = "eager"
    model = "eager/model"

    def generate(self, prompt: str, *, json_mode: bool = False) -> LLMResult:
        import json as _json

        return LLMResult(
            text=_json.dumps(
                {
                    "response": "Please restart the device and let us know.",
                    "should_escalate": False,
                    "escalation_reason": "",
                    "evidence_ids": ["E1"],
                    "confidence": 1.0,
                }
            ),
            provider=self.name,
            model=self.model,
            prompt_tokens=10,
            completion_tokens=10,
        )

    @property
    def totals(self) -> dict:
        return {"requests": 1, "failures": 0, "cost_usd": 0.0}


def agent_for(intent, confidence, security, context=True, retrieval=0.99):
    return ReplyAgent(
        classifier=FixedClassifier(intent, confidence, security, context),
        retriever=FixedRetriever(retrieval),
        generator=StructuredLLMGenerator(EagerProvider()),
    )


class TestSecurityEscalationCannotBeDowngraded:
    @pytest.mark.parametrize("intent", sorted(TAXONOMY.names))
    def test_every_intent_escalates_when_security_sensitive(self, intent):
        decision = agent_for(intent, 0.99, security=True).handle("something happened")
        assert decision.action == "ESCALATE"
        assert decision.reason is EscalationReason.SECURITY_SENSITIVE

    @pytest.mark.parametrize("confidence", CONFIDENCES)
    def test_no_confidence_value_clears_the_security_gate(self, confidence):
        decision = agent_for("battery_charging", confidence, security=True).handle("hi")
        assert decision.action == "ESCALATE"
        assert decision.reason is EscalationReason.SECURITY_SENSITIVE

    @pytest.mark.parametrize("retrieval", [0.0, 0.5, 0.9, 1.0])
    def test_no_retrieval_score_clears_the_security_gate(self, retrieval):
        decision = agent_for(
            "battery_charging", 0.99, security=True, retrieval=retrieval
        ).handle("hi")
        assert decision.reason is EscalationReason.SECURITY_SENSITIVE

    def test_an_eager_model_cannot_talk_routing_out_of_escalating(self):
        # The model says should_escalate=false at confidence 1.0 with a clean reply.
        decision = agent_for("battery_charging", 0.99, security=True).handle("hi")
        assert decision.action == "ESCALATE"
        assert decision.reply is None

    def test_the_generator_is_never_even_consulted(self):
        # Cheapest possible proof: a security-flagged message must not reach a paid call.
        provider = EagerProvider()
        calls = []
        original = provider.generate
        provider.generate = lambda *a, **k: (calls.append(1), original(*a, **k))[1]
        agent = ReplyAgent(
            classifier=FixedClassifier("battery_charging", 0.99, True),
            retriever=FixedRetriever(),
            generator=StructuredLLMGenerator(provider),
        )
        agent.handle("someone took over my account")
        assert calls == []

    def test_no_reply_text_is_ever_emitted_on_a_security_escalation(self):
        for intent in sorted(TAXONOMY.names):
            assert agent_for(intent, 1.0, security=True).handle("hi").reply is None

    def test_security_outranks_insufficient_context_in_the_reason(self):
        # Both gates fire; the reason recorded must be the safety-critical one, because the
        # escalation reason is what a human triages on.
        decision = agent_for("other_unclear", 0.0, security=True, context=False).handle("ok")
        assert decision.reason is EscalationReason.SECURITY_SENSITIVE


class TestTheHappyPathStillExists:
    """A gate that escalates everything is not a safety feature, it is an outage."""

    def test_a_safe_well_evidenced_message_is_still_auto_handled(self):
        decision = agent_for("battery_charging", 0.9, security=False).handle(
            "my battery drains quickly since yesterday"
        )
        assert decision.action == "AUTO_HANDLE"
        assert decision.reply

    @pytest.mark.parametrize("confidence", CONFIDENCES)
    def test_confidence_does_not_change_routing_for_a_safe_message(self, confidence):
        decision = agent_for("battery_charging", confidence, security=False).handle(
            "my battery drains quickly since yesterday"
        )
        assert decision.action == "AUTO_HANDLE"


class TestPolicyAndRelevanceGatesAreWired:
    def test_a_deflecting_reply_escalates_even_though_it_is_grounded(self):
        class Deflecting(EagerProvider):
            def generate(self, prompt, *, json_mode=False):
                import json as _json

                return LLMResult(
                    text=_json.dumps(
                        {
                            "response": "Please DM us with your serial number.",
                            "should_escalate": False,
                            "escalation_reason": "",
                            "evidence_ids": ["E1"],
                            "confidence": 0.9,
                        }
                    ),
                    provider="x",
                    model="y",
                )

        agent = ReplyAgent(
            classifier=FixedClassifier("battery_charging", 0.9, False),
            retriever=FixedRetriever(),
            generator=StructuredLLMGenerator(Deflecting()),
        )
        decision = agent.handle("my battery drains quickly")
        assert decision.action == "ESCALATE"
        assert decision.reason is EscalationReason.POLICY_VIOLATION

    def test_contradictory_evidence_escalates_before_a_model_is_called(self):
        class Counting(EagerProvider):
            calls = 0

            def generate(self, prompt, *, json_mode=False):
                Counting.calls += 1
                return super().generate(prompt, json_mode=json_mode)

        contradicting = (
            EvidenceCase(
                case_id="case0",
                customer_text="autocorrect is broken",
                resolution_text="We have released 11.0.2. Make sure you back up and get updated.",
                created_at=WHEN - timedelta(days=1),
                score=0.9,
            ),
        )

        class Retriever:
            def retrieve(self, query, top_k=4, exclude_ids=None, before=None):
                return RetrievalResult(query=query, cases=contradicting, confidence=0.9)

        agent = ReplyAgent(
            classifier=FixedClassifier("device_malfunction", 0.9, False),
            retriever=Retriever(),
            generator=StructuredLLMGenerator(Counting()),
        )
        decision = agent.handle("my keyboard broke ever since 11.0.2, when is the fix")
        assert decision.action == "ESCALATE"
        assert decision.reason is EscalationReason.CONTRADICTORY_EVIDENCE
        assert Counting.calls == 0, "a model was paid for evidence already known to be wrong"
