from __future__ import annotations

from drugclaw.agent_reflector import ReflectorAgent
from drugclaw.models import AgentState


class _LLMStub:
    def __init__(self, payload):
        self.payload = payload

    def generate_json(self, messages, temperature=0.3):
        return dict(self.payload)


class _ConfigStub:
    def __init__(self, max_iterations: int):
        self.MAX_ITERATIONS = max_iterations
        self.EVIDENCE_THRESHOLD_EPSILON = 0.1


def test_reflector_allows_first_iteration_when_max_iterations_is_zero() -> None:
    reflector = ReflectorAgent(
        _LLMStub(
            {
                "evidence_sufficiency_score": 0.3,
                "current_reward": 0.6,
                "evidence_sufficient": False,
                "should_continue": True,
                "reasoning": "Need more evidence.",
            }
        ),
        _ConfigStub(max_iterations=0),
    )
    state = AgentState(original_query="q")

    reflected = reflector.execute(state)

    assert reflected.should_continue is True
    assert reflected.max_iterations_reached is False


def test_reflector_allows_one_fallback_before_max_iterations_limit() -> None:
    reflector = ReflectorAgent(
        _LLMStub(
            {
                "evidence_sufficiency_score": 0.3,
                "current_reward": 0.6,
                "evidence_sufficient": False,
                "should_continue": True,
                "reasoning": "Need one more evidence pass.",
            }
        ),
        _ConfigStub(max_iterations=2),
    )
    state = AgentState(original_query="q", iteration=1)

    reflected = reflector.execute(state)

    assert reflected.should_continue is True
    assert reflected.max_iterations_reached is False


def test_reflector_marks_limit_reached_when_iteration_hits_max() -> None:
    reflector = ReflectorAgent(
        _LLMStub(
            {
                "evidence_sufficiency_score": 0.3,
                "current_reward": 0.6,
                "evidence_sufficient": False,
                "should_continue": True,
                "reasoning": "Need more evidence.",
            }
        ),
        _ConfigStub(max_iterations=2),
    )
    state = AgentState(original_query="q", iteration=2)

    reflected = reflector.execute(state)

    assert reflected.should_continue is False
    assert reflected.max_iterations_reached is True
