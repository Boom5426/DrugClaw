from __future__ import annotations

from drugclaw.agent_websearch import WebSearchAgent
from drugclaw.models import AgentState
from drugclaw.query_plan import QueryPlan
from drugclaw.skills.base import RetrievalResult
from drugclaw.skills.web_search.web_search.web_search_skill import WebSearchSkill


class _WebSkillStub:
    def search(self, query: str, max_results: int = 10):
        return [
            RetrievalResult(
                source_entity=query,
                source_type="query",
                target_entity="Clopidogrel and CYP2C19",
                target_type="publication",
                relationship="literature_result",
                weight=1.0,
                source="PubMed",
                evidence_text="CPIC guidance discusses CYP2C19 and clopidogrel response.",
                sources=["https://pubmed.ncbi.nlm.nih.gov/12345678/"],
                metadata={"url": "https://pubmed.ncbi.nlm.nih.gov/12345678/"},
            ),
            RetrievalResult(
                source_entity=query,
                source_type="query",
                target_entity="General blog summary",
                target_type="web_page",
                relationship="web_result",
                weight=0.8,
                source="DuckDuckGo",
                evidence_text="A general blog mentions clopidogrel metabolism.",
                sources=["https://example.com/clopidogrel-blog"],
                metadata={"url": "https://example.com/clopidogrel-blog"},
            ),
        ]


class _LowAuthorityWebSkillStub:
    def search(self, query: str, max_results: int = 10):
        return [
            RetrievalResult(
                source_entity=query,
                source_type="query",
                target_entity="Google Maps result",
                target_type="web_page",
                relationship="web_result",
                weight=1.0,
                source="DuckDuckGo",
                evidence_text="A generic maps result unrelated to pharmacogenomic guidance.",
                sources=["https://maps.google.co.jp/mapfiles/home3.html"],
                metadata={"url": "https://maps.google.co.jp/mapfiles/home3.html"},
            )
        ]


class _SourceAwareWebSkillStub:
    def __init__(self):
        self.calls = []

    def search_with_source(self, query: str, *, source: str | None = None, max_results: int = 10):
        self.calls.append((query, source, max_results))
        if source == "pubmed" and query == "What pharmacogenomic factors affect clopidogrel efficacy and safety?":
            return [
                RetrievalResult(
                    source_entity=query,
                    source_type="query",
                    target_entity="CPIC-guided clopidogrel therapy",
                    target_type="publication",
                    relationship="literature_result",
                    weight=1.0,
                    source="PubMed",
                    evidence_text="Clinical guidance supports altered antiplatelet strategy in CYP2C19 poor metabolizers.",
                    sources=["https://pubmed.ncbi.nlm.nih.gov/12345678/"],
                    metadata={"url": "https://pubmed.ncbi.nlm.nih.gov/12345678/"},
                )
            ]
        return []


def test_websearch_agent_simple_lane_filters_to_authority_sources_for_high_risk_query() -> None:
    agent = WebSearchAgent(llm_client=object(), web_search_skill=_WebSkillStub())
    state = AgentState(
        original_query="What pharmacogenomic factors affect clopidogrel efficacy and safety?",
        query_plan=QueryPlan(
            question_type="pharmacogenomics",
            entities={"drug": ["clopidogrel"]},
            subquestions=[],
            preferred_skills=["PharmGKB", "CPIC"],
            preferred_evidence_types=[],
            requires_graph_reasoning=False,
            requires_prediction_sources=False,
            requires_web_fallback=False,
            answer_risk_level="high",
            notes=[],
        ),
    )

    updated = agent.execute_simple(state)

    assert len(updated.web_search_results) == 1
    assert updated.web_search_results[0]["source"] == "PubMed"


def test_websearch_agent_simple_lane_drops_low_authority_fallbacks_for_high_risk_query() -> None:
    agent = WebSearchAgent(llm_client=object(), web_search_skill=_LowAuthorityWebSkillStub())
    state = AgentState(
        original_query="What pharmacogenomic factors affect clopidogrel efficacy and safety?",
        query_plan=QueryPlan(
            question_type="pharmacogenomics",
            entities={"drug": ["clopidogrel"]},
            subquestions=[],
            preferred_skills=["PharmGKB", "CPIC"],
            preferred_evidence_types=[],
            requires_graph_reasoning=False,
            requires_prediction_sources=False,
            requires_web_fallback=False,
            answer_risk_level="high",
            notes=[],
        ),
    )

    updated = agent.execute_simple(state)

    assert updated.web_search_results == []
    assert "Web Evidence:" not in updated.retrieved_text


def test_websearch_agent_simple_lane_honors_pubmed_source_hint_with_websearch_skill(
    monkeypatch,
) -> None:
    skill = WebSearchSkill({})

    monkeypatch.setattr(skill, "_ddg_search", lambda query, max_results, timeout: [])

    def _pubmed_search(query: str, max_results: int, timeout: int):
        return [
            RetrievalResult(
                source_entity=query,
                source_type="query",
                target_entity="CPIC-guided clopidogrel therapy",
                target_type="publication",
                relationship="literature_result",
                weight=1.0,
                source="PubMed",
                evidence_text="Clinical guidance supports altered antiplatelet strategy in CYP2C19 poor metabolizers.",
                sources=["https://pubmed.ncbi.nlm.nih.gov/12345678/"],
                metadata={"url": "https://pubmed.ncbi.nlm.nih.gov/12345678/"},
            )
        ]

    monkeypatch.setattr(skill, "_pubmed_search", _pubmed_search)

    agent = WebSearchAgent(llm_client=object(), web_search_skill=skill)
    state = AgentState(
        original_query="What pharmacogenomic factors affect clopidogrel efficacy and safety?",
        query_plan=QueryPlan(
            question_type="pharmacogenomics",
            entities={"drug": ["clopidogrel"]},
            subquestions=[],
            preferred_skills=["PharmGKB", "CPIC"],
            preferred_evidence_types=[],
            requires_graph_reasoning=False,
            requires_prediction_sources=False,
            requires_web_fallback=False,
            answer_risk_level="high",
            notes=[],
        ),
    )

    updated = agent.execute_simple(state)

    assert len(updated.web_search_results) == 1
    assert updated.web_search_results[0]["source"] == "PubMed"
    assert "Web Evidence:" in updated.retrieved_text


def test_websearch_agent_simple_lane_uses_original_query_for_pubmed_pgx_search() -> None:
    skill = _SourceAwareWebSkillStub()
    agent = WebSearchAgent(llm_client=object(), web_search_skill=skill)
    state = AgentState(
        original_query="What pharmacogenomic factors affect clopidogrel efficacy and safety?",
        query_plan=QueryPlan(
            question_type="pharmacogenomics",
            entities={"drug": ["clopidogrel"]},
            subquestions=[],
            preferred_skills=["PharmGKB", "CPIC"],
            preferred_evidence_types=[],
            requires_graph_reasoning=False,
            requires_prediction_sources=False,
            requires_web_fallback=False,
            answer_risk_level="high",
            notes=[],
        ),
    )

    updated = agent.execute_simple(state)

    assert skill.calls[0][0] == "What pharmacogenomic factors affect clopidogrel efficacy and safety?"
    assert skill.calls[0][1] == "pubmed"
    assert len(updated.web_search_results) == 1
