from __future__ import annotations

from pathlib import Path

from drugclaw.knowhow_registry import KnowHowRegistry


def test_default_knowhow_registry_loads_seed_documents() -> None:
    registry = KnowHowRegistry()

    direct_targets = registry.get_document("direct_targets_grounding")
    mechanism = registry.get_document("mechanism_explanation")
    repurposing = registry.get_document("repurposing_evidence_thresholds")
    ddi = registry.get_document("clinical_ddi_prioritization")
    pgx = registry.get_document("pgx_guidance_prioritization")
    labeling = registry.get_document("labeling_regulatory_priority")
    adr = registry.get_document("adr_signal_tiering")

    assert direct_targets is not None
    assert mechanism is not None
    assert repurposing is not None
    assert ddi is not None
    assert pgx is not None
    assert labeling is not None
    assert adr is not None

    assert direct_targets.task_types == ["direct_targets", "target_profile"]
    assert mechanism.task_types == ["mechanism_of_action"]
    assert repurposing.task_types == ["repurposing_evidence"]
    assert ddi.task_types == ["clinically_relevant_ddi", "ddi_mechanism"]
    assert pgx.task_types == ["pgx_guidance"]
    assert labeling.task_types == ["labeling_summary"]
    assert adr.task_types == ["major_adrs"]

    for document in (direct_targets, mechanism, repurposing, ddi, pgx, labeling, adr):
        assert document.body_path
        assert Path(document.body_path).exists()

    assert direct_targets.declared_by_skills == [
        "BindingDB",
        "ChEMBL",
        "DGIdb",
        "DrugBank",
        "Open Targets Platform",
        "STITCH",
        "TarKG",
    ]
    assert mechanism.declared_by_skills == ["DRUGMECHDB", "Open Targets Platform"]
    assert repurposing.declared_by_skills == ["DrugBank", "DrugCentral", "RepoDB"]
    assert ddi.declared_by_skills == ["DDInter", "KEGG Drug", "MecDDI"]
    assert pgx.declared_by_skills == ["CPIC", "PharmGKB"]
    assert labeling.declared_by_skills == [
        "DailyMed",
        "MedlinePlus Drug Info",
        "openFDA Human Drug",
    ]
    assert adr.declared_by_skills == ["ADReCS", "FAERS", "nSIDES", "SIDER"]


def test_knowhow_registry_skips_examples_directory(tmp_path: Path) -> None:
    knowhow_dir = tmp_path / "resources_metadata" / "knowhow"
    examples_dir = knowhow_dir / "examples"
    live_dir = knowhow_dir / "dti"
    examples_dir.mkdir(parents=True, exist_ok=True)
    live_dir.mkdir(parents=True, exist_ok=True)

    (live_dir / "live_doc.json").write_text(
        """
        {
          "doc_id": "live_doc",
          "title": "Live doc",
          "task_types": ["direct_targets"],
          "evidence_types": ["database_record"],
          "risk_level": "medium",
          "conflict_policy": "Prefer direct evidence.",
          "answer_template": "direct_targets",
          "max_prompt_snippets": 1
        }
        """.strip(),
        encoding="utf-8",
    )
    (live_dir / "live_doc.md").write_text("Live body\n", encoding="utf-8")
    (examples_dir / "minimal_knowhow.json").write_text(
        """
        {
          "doc_id": "example_doc",
          "title": "Example doc",
          "task_types": ["direct_targets"],
          "evidence_types": ["database_record"],
          "risk_level": "medium",
          "conflict_policy": "Example only.",
          "answer_template": "direct_targets",
          "max_prompt_snippets": 1
        }
        """.strip(),
        encoding="utf-8",
    )
    (examples_dir / "minimal_knowhow.md").write_text("Example body\n", encoding="utf-8")

    registry = KnowHowRegistry(repo_root=tmp_path)

    assert registry.get_document("live_doc") is not None
    assert registry.get_document("example_doc") is None
