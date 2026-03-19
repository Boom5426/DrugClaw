"""Tests for the EntityResolver fuzzy matching and variant expansion."""
from __future__ import annotations

from drugclaw.entity_resolver import EntityResolver


# ---------------------------------------------------------------------------
# Stubs
# ---------------------------------------------------------------------------

class _FakeSkill:
    """Minimal skill stub for testing local index building."""

    def __init__(self, name: str, access_mode: str, drug_index=None, rows=None):
        self.name = name
        self.access_mode = access_mode
        self._drug_index = drug_index or {}
        self._gene_index = {}
        self._rows = rows or []

    def is_available(self):
        return True


class _FakeRegistry:
    """Minimal SkillRegistry stub."""

    def __init__(self, skills: dict[str, _FakeSkill]):
        self._skills = skills

    def list_by_access_mode(self, mode: str) -> list[str]:
        return [n for n, s in self._skills.items() if s.access_mode == mode]

    def get_skill(self, name: str):
        return self._skills.get(name)


class _LLMStub:
    """Stub LLM that returns canned variant responses."""

    def __init__(self, variants: list[str] | None = None):
        self.variants = variants or []
        self.call_count = 0

    def generate_json(self, messages, temperature=0.1):
        self.call_count += 1
        return {"variants": self.variants}


# ---------------------------------------------------------------------------
# Tests: basic normalisation
# ---------------------------------------------------------------------------

def test_resolve_returns_original_entities_unchanged_when_no_matches():
    resolver = EntityResolver()
    entities = {"drug": ["imatinib"]}
    result = resolver.resolve(entities, skill_names=[], use_llm=False)
    assert result == {"drug": ["imatinib"]}


def test_resolve_deduplicates_case_insensitive():
    resolver = EntityResolver()
    entities = {"drug": ["Imatinib", "imatinib", "IMATINIB"]}
    result = resolver.resolve(entities, skill_names=[], use_llm=False)
    assert len(result["drug"]) == 1
    assert result["drug"][0] == "Imatinib"  # first occurrence wins


def test_resolve_preserves_non_resolvable_types():
    resolver = EntityResolver()
    entities = {"other": ["something"]}
    result = resolver.resolve(entities, skill_names=[], use_llm=False)
    assert result == {"other": ["something"]}


# ---------------------------------------------------------------------------
# Tests: local fuzzy matching
# ---------------------------------------------------------------------------

def _make_local_resolver(drug_names: list[str]) -> tuple[EntityResolver, _FakeRegistry]:
    skill = _FakeSkill(
        name="TestLocal",
        access_mode="LOCAL_FILE",
        drug_index={name.lower(): [] for name in drug_names},
    )
    registry = _FakeRegistry({"TestLocal": skill})
    resolver = EntityResolver(skill_registry=registry)
    return resolver, registry


def test_fuzzy_match_corrects_typo():
    resolver, _ = _make_local_resolver(["imatinib", "aspirin", "metformin"])
    entities = {"drug": ["imatanib"]}  # typo: a instead of i
    result = resolver.resolve(entities, skill_names=["TestLocal"], use_llm=False)
    assert "imatinib" in [d.lower() for d in result["drug"]]


def test_fuzzy_match_handles_case_insensitive():
    resolver, _ = _make_local_resolver(["Imatinib", "Aspirin"])
    entities = {"drug": ["IMATINIB"]}
    result = resolver.resolve(entities, skill_names=["TestLocal"], use_llm=False)
    # Original is kept; no extra fuzzy match needed since it's case-identical
    assert len(result["drug"]) >= 1
    assert result["drug"][0] == "IMATINIB"


def test_fuzzy_match_no_false_positives_for_unrelated():
    resolver, _ = _make_local_resolver(["imatinib", "aspirin"])
    entities = {"drug": ["zzzzzzz"]}
    result = resolver.resolve(entities, skill_names=["TestLocal"], use_llm=False)
    # Should not match anything
    assert result["drug"] == ["zzzzzzz"]


def test_fuzzy_match_from_row_data():
    """Test index building from _rows pattern (not pre-built index)."""
    skill = _FakeSkill(
        name="RowSkill",
        access_mode="LOCAL_FILE",
        rows=[
            {"drug": "Sorafenib", "target": "BRAF"},
            {"drug": "Erlotinib", "target": "EGFR"},
            {"drug": "Gefitinib", "target": "EGFR"},
        ],
    )
    registry = _FakeRegistry({"RowSkill": skill})
    resolver = EntityResolver(skill_registry=registry)
    entities = {"drug": ["sorafanib"]}  # typo
    result = resolver.resolve(entities, skill_names=["RowSkill"], use_llm=False)
    drug_lower = [d.lower() for d in result["drug"]]
    assert "sorafenib" in drug_lower


# ---------------------------------------------------------------------------
# Tests: LLM variant generation
# ---------------------------------------------------------------------------

def test_llm_variants_added_for_api_skills():
    api_skill = _FakeSkill(name="ChEMBL", access_mode="REST_API")
    registry = _FakeRegistry({"ChEMBL": api_skill})
    llm = _LLMStub(variants=["Gleevec", "STI-571"])
    resolver = EntityResolver(skill_registry=registry, llm_client=llm)

    entities = {"drug": ["imatinib"]}
    result = resolver.resolve(entities, skill_names=["ChEMBL"], use_llm=True)

    assert "Gleevec" in result["drug"]
    assert "STI-571" in result["drug"]
    assert result["drug"][0] == "imatinib"  # original first
    assert llm.call_count == 1


def test_llm_not_called_when_use_llm_false():
    api_skill = _FakeSkill(name="ChEMBL", access_mode="REST_API")
    registry = _FakeRegistry({"ChEMBL": api_skill})
    llm = _LLMStub(variants=["Gleevec"])
    resolver = EntityResolver(skill_registry=registry, llm_client=llm)

    entities = {"drug": ["imatinib"]}
    result = resolver.resolve(entities, skill_names=["ChEMBL"], use_llm=False)

    assert llm.call_count == 0
    assert result["drug"] == ["imatinib"]


def test_llm_variants_deduplicated_against_originals():
    api_skill = _FakeSkill(name="ChEMBL", access_mode="REST_API")
    registry = _FakeRegistry({"ChEMBL": api_skill})
    llm = _LLMStub(variants=["imatinib", "Imatinib", "gleevec"])  # first two are dupes
    resolver = EntityResolver(skill_registry=registry, llm_client=llm)

    entities = {"drug": ["imatinib"]}
    result = resolver.resolve(entities, skill_names=["ChEMBL"], use_llm=True)

    assert result["drug"].count("imatinib") == 1
    assert "gleevec" in [d.lower() for d in result["drug"]]


def test_llm_error_handled_gracefully():
    api_skill = _FakeSkill(name="ChEMBL", access_mode="REST_API")
    registry = _FakeRegistry({"ChEMBL": api_skill})

    class _ErrorLLM:
        def generate_json(self, *args, **kwargs):
            raise RuntimeError("LLM unavailable")

    resolver = EntityResolver(skill_registry=registry, llm_client=_ErrorLLM())
    entities = {"drug": ["imatinib"]}
    result = resolver.resolve(entities, skill_names=["ChEMBL"], use_llm=True)

    # Should still return originals without crashing
    assert result["drug"] == ["imatinib"]


# ---------------------------------------------------------------------------
# Tests: mixed local + API scenario
# ---------------------------------------------------------------------------

def test_mixed_local_and_api_skills():
    local_skill = _FakeSkill(
        name="TTD",
        access_mode="LOCAL_FILE",
        drug_index={"imatinib": [], "sorafenib": [], "erlotinib": []},
    )
    api_skill = _FakeSkill(name="ChEMBL", access_mode="REST_API")
    registry = _FakeRegistry({"TTD": local_skill, "ChEMBL": api_skill})
    llm = _LLMStub(variants=["Gleevec"])
    resolver = EntityResolver(skill_registry=registry, llm_client=llm)

    entities = {"drug": ["imatanib"]}  # typo
    result = resolver.resolve(
        entities, skill_names=["TTD", "ChEMBL"], use_llm=True,
    )

    drug_lower = [d.lower() for d in result["drug"]]
    # Should have: original, fuzzy local match, LLM variant
    assert "imatanib" in drug_lower  # original preserved
    assert "imatinib" in drug_lower  # fuzzy match
    assert "gleevec" in drug_lower   # LLM variant


# ---------------------------------------------------------------------------
# Tests: empty / edge cases
# ---------------------------------------------------------------------------

def test_resolve_empty_entities():
    resolver = EntityResolver()
    assert resolver.resolve({}, skill_names=[]) == {}


def test_resolve_empty_entity_list():
    resolver = EntityResolver()
    result = resolver.resolve({"drug": []}, skill_names=[], use_llm=False)
    assert result == {"drug": []}


def test_build_local_index_with_no_registry():
    resolver = EntityResolver()
    resolver.build_local_index()
    assert resolver._local_index == {}
