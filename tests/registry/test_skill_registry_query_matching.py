from __future__ import annotations

from drugclaw.skills.base import RAGSkill
from drugclaw.skills.registry import SkillRegistry


class _StubSkill(RAGSkill):
    _implemented = True

    def __init__(
        self,
        *,
        name: str,
        subcategory: str,
        aim: str,
        data_range: str = "",
        access_mode: str = "REST_API",
    ):
        super().__init__()
        self.name = name
        self.subcategory = subcategory
        self.aim = aim
        self.data_range = data_range
        self.access_mode = access_mode

    def retrieve(self, entities, query: str = "", max_results: int = 50, **kwargs):
        return []


def test_skill_registry_matches_target_queries_with_punctuation() -> None:
    registry = SkillRegistry()
    registry.register(
        _StubSkill(
            name="ChEMBL",
            subcategory="dti",
            aim="drug target interactions and protein targets",
        )
    )
    registry.register(
        _StubSkill(
            name="DailyMed",
            subcategory="drug_labeling",
            aim="drug labeling and prescribing information",
        )
    )

    suggestions = registry.get_skills_for_query("What does imatinib target?")

    assert suggestions == ["ChEMBL"]
