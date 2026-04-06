from __future__ import annotations

from .eval_models import EvalExpectation, EvalTaskCase


def build_cli_usability_min_pack() -> list[EvalTaskCase]:
    required_metadata_keys = [
        "git_sha",
        "model",
        "base_url",
        "doctor_summary",
        "network_notes",
        "suite_name",
    ]
    return [
        EvalTaskCase(
            task_id="cli_usability::target_moa_imatinib",
            dataset_name="cli_usability::target_moa_imatinib",
            plan_type="single_task",
            task_type="mechanism",
            legacy_question_type="mechanism",
            query="What are the known drug targets and mechanism of action of imatinib?",
            expectation=EvalExpectation(
                scorer="cli_usability_contract",
                expected_resources=["Open Targets Platform", "DRUGMECHDB", "BindingDB", "ChEMBL"],
                must_hit_patterns=[
                    ["abl1", "bcr-abl"],
                    ["kit", "pdgfr", "pdgfra", "pdgfrb"],
                    ["tyrosine kinase inhibitor", "kinase inhibitor", "tyrosine-kinase inhibitor"],
                ],
                hard_fail_patterns=[],
                rerun_policy="on_demand",
                required_metadata_keys=required_metadata_keys,
            ),
        ),
        EvalTaskCase(
            task_id="cli_usability::repurposing_metformin",
            dataset_name="cli_usability::repurposing_metformin",
            plan_type="single_task",
            task_type="repurposing_evidence",
            legacy_question_type="drug_repurposing",
            query="What are the approved indications and repurposing evidence of metformin?",
            expectation=EvalExpectation(
                scorer="cli_usability_contract",
                expected_resources=["RepoDB", "DrugCentral", "DrugBank"],
                must_hit_patterns=[
                    ["type 2 diabetes", "type 2 diabetes mellitus", "type ii diabetes"],
                    ["repurposing", "repositioning", "exploratory evidence", "limited evidence", "evidence gap", "off-label"],
                ],
                hard_fail_patterns=["has_warning", "has_adverse_reaction", "structured repurposing evidence"],
                rerun_policy="fixed",
                required_metadata_keys=required_metadata_keys,
            ),
        ),
        EvalTaskCase(
            task_id="cli_usability::safety_clozapine",
            dataset_name="cli_usability::safety_clozapine",
            plan_type="single_task",
            task_type="major_adrs",
            legacy_question_type="adr",
            query="What are the major safety risks and serious adverse reactions of clozapine?",
            expectation=EvalExpectation(
                scorer="cli_usability_contract",
                expected_resources=["ADReCS", "FAERS", "nSIDES", "SIDER"],
                must_hit_patterns=[
                    ["agranulocytosis", "severe neutropenia"],
                    ["anc", "cbc"],
                    ["myocarditis", "cardiomyopathy", "seizures", "severe gi hypomotility"],
                ],
                hard_fail_patterns=["no structured evidence was retrieved"],
                rerun_policy="fixed",
                required_metadata_keys=required_metadata_keys,
            ),
        ),
        EvalTaskCase(
            task_id="cli_usability::ddi_warfarin",
            dataset_name="cli_usability::ddi_warfarin",
            plan_type="single_task",
            task_type="ddi_mechanism",
            legacy_question_type="ddi_mechanism",
            query="What are the clinically important drug-drug interactions of warfarin and their mechanisms?",
            expectation=EvalExpectation(
                scorer="cli_usability_contract",
                expected_resources=["DDInter", "KEGG Drug", "MecDDI"],
                must_hit_patterns=[
                    ["aspirin", "ibuprofen", "nsaids", "antiplatelet"],
                    ["pharmacodynamic", "cyp", "metabolism", "mechanism"],
                    ["inr", "bleeding monitoring", "monitoring"],
                ],
                hard_fail_patterns=[],
                rerun_policy="on_demand",
                required_metadata_keys=required_metadata_keys,
            ),
        ),
        EvalTaskCase(
            task_id="cli_usability::labeling_metformin",
            dataset_name="cli_usability::labeling_metformin",
            plan_type="single_task",
            task_type="labeling_summary",
            legacy_question_type="labeling",
            query="What key prescribing and clinical use information should be considered for metformin?",
            expectation=EvalExpectation(
                scorer="cli_usability_contract",
                expected_resources=["DailyMed", "openFDA Human Drug", "MedlinePlus Drug Info"],
                must_hit_patterns=[
                    ["type 2 diabetes", "approved use", "glycemic control"],
                    ["renal", "egfr", "kidney"],
                    ["monitor", "lactic acidosis", "hold", "temporary discontinue"],
                ],
                hard_fail_patterns=["has_warning", "has_adverse_reaction"],
                rerun_policy="fixed",
                required_metadata_keys=required_metadata_keys,
            ),
        ),
        EvalTaskCase(
            task_id="cli_usability::pgx_clopidogrel",
            dataset_name="cli_usability::pgx_clopidogrel",
            plan_type="single_task",
            task_type="pgx_guidance",
            legacy_question_type="pharmacogenomics",
            query="What pharmacogenomic factors affect clopidogrel efficacy and safety?",
            expectation=EvalExpectation(
                scorer="cli_usability_contract",
                expected_resources=["PharmGKB", "CPIC"],
                must_hit_patterns=[
                    ["cyp2c19"],
                    ["reduced activation", "loss-of-function", "poor metabolizer", "active metabolite"],
                    ["alternative", "prasugrel", "ticagrelor", "guidance", "cpic"],
                ],
                hard_fail_patterns=[],
                rerun_policy="on_demand",
                required_metadata_keys=required_metadata_keys,
            ),
        ),
    ]
