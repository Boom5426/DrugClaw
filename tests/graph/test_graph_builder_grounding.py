from __future__ import annotations

from drugclaw.agent_graph_builder import GraphBuilderAgent
from drugclaw.evidence import EvidenceItem
from drugclaw.models import AgentState


class _LLMStub:
    def generate_json(self, messages, temperature=0.3):
        return {
            "triples": [
                {
                    "source_entity": "Imatinib",
                    "source_type": "drug",
                    "target_entity": "ABL1",
                    "target_type": "gene",
                    "relationship": "targets",
                    "confidence": 0.9,
                    "evidence_text": "BindingDB reports imatinib against ABL1.",
                    "source_db": "BindingDB",
                    "evidence_ids": ["E1"],
                },
                {
                    "source_entity": "Imatinib",
                    "source_type": "drug",
                    "target_entity": "KIT",
                    "target_type": "gene",
                    "relationship": "may_target",
                    "confidence": 0.8,
                    "evidence_text": "Possible secondary target.",
                    "source_db": "SpeculativeModel",
                    "evidence_ids": [],
                },
            ]
        }


def test_graph_builder_marks_edges_without_evidence_as_low_confidence() -> None:
    state = AgentState(original_query="What does imatinib target?")
    state.retrieved_text = "Structured evidence payload"
    state.evidence_items = [
        EvidenceItem(
            evidence_id="E1",
            source_skill="BindingDB",
            source_type="database",
            source_title="BindingDB binding record",
            source_locator="PMID:12345678",
            snippet="Imatinib Ki=21 nM against ABL1.",
            structured_payload={},
            claim="Imatinib targets ABL1.",
            evidence_kind="database_record",
            support_direction="supports",
            confidence=0.0,
            retrieval_score=0.92,
            timestamp="2026-03-18T00:00:00Z",
            metadata={},
        )
    ]

    updated = GraphBuilderAgent(_LLMStub()).execute(state)

    grounded = next(edge for edge in updated.current_subgraph.edges if edge.target.name == "ABL1")
    ungrounded = next(edge for edge in updated.current_subgraph.edges if edge.target.name == "KIT")

    assert grounded.attributes["evidence_ids"] == ["E1"]
    assert grounded.attributes["evidence_support"] == "grounded"
    assert ungrounded.attributes["evidence_support"] == "ungrounded"
    assert ungrounded.confidence <= 0.25
