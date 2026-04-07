"""
DrugClaw RAG skills package.

The runtime resource registry is the authoritative source for active resource
counts and status. This package keeps the concrete skill implementations,
their catalog metadata, and the build_default_registry() factory.
"""
from .base import RAGSkill, DatasetRAGSkill, RetrievalResult, AccessMode
from .registry import SkillRegistry
from .skill_tree import SkillTree, SkillNode, Subcategory
try:
    from ..resource_path_resolver import resolve_skill_config_paths
except ImportError:  # pragma: no cover - compatibility for top-level `skills` import
    from drugclaw.resource_path_resolver import resolve_skill_config_paths

SubDomain = Subcategory  # legacy alias
Domain = Subcategory     # legacy alias

# --- Implemented skills — have example.py + SKILL.md ---
from .dti import (ChEMBLSkill, BindingDBSkill, DGIdbSkill, OpenTargetsSkill,
    TTDSkill, STITCHSkill, TarKGSkill, GDKDSkill, MolecularTargetsSkill,
    MolecularTargetsDataSkill)
from .adr import (FAERSSkill, SIDERSkill, nSIDESSkill, ADReCSSkill)
from .drug_knowledgebase import (UniD3Skill, DrugBankSkill, IUPHARSkill,
    DrugCentralSkill, CPICSkill, PharmKGSkill, WHOEssentialMedicinesSkill,
    FDAOrangeBookSkill)
from .drug_mechanism import DrugMechDBSkill
from .drug_labeling import (OpenFDASkill, DailyMedSkill, MedlinePlusSkill)
from .drug_ontology import (RxNormSkill, ChEBISkill, ATCSkill, NDFRTSkill)
from .drug_repurposing import (RepoDBSkill, DRKGSkill, OREGANOSkill,
    RepurposingHubSkill, DrugRepoBankSkill, RepurposeDrugsSkill)
from .pharmacogenomics import PharmGKBSkill
from .ddi import (MecDDISkill, DDInterSkill, KEGGDrugSkill)
from .drug_review import (WebMDReviewsSkill, DrugsComReviewsSkill)
from .drug_toxicity import (UniToxSkill, LiverToxSkill, DILIrankSkill, DILISkill)
from .drug_combination import (DrugCombDBSkill, DrugCombSkill)
from .drug_molecular_property import GDSCSkill
from .drug_disease import SemaTyPSkill
from .drug_nlp import (DDICorpusSkill, DrugProtSkill, ADECorpusSkill,
    CADECSkill, PsyTARSkill, TAC2017ADRSkill, PHEESkill)
from .web_search import WebSearchSkill

# --- Stub skills (13) — interface preserved, not registered ---
from .dti import (PROMISCUOUSSkill, DTCSkill)
from .adr import VigiAccessSkill
from .drug_knowledgebase import DrugsComSkill
from .drug_labeling import RxListSkill
from .drug_repurposing import (DrugRepurposingOnlineSkill,
    CancerDRSkill, EKDRDSkill)
from .drug_combination import (CDCDBSkill, DCDBSkill)
from .drug_nlp import (DrugEHRQASkill, N2C22018Skill)
from .drug_review import AskAPatientSkill


def build_default_registry(config) -> SkillRegistry:
    """
    Build a SkillRegistry pre-loaded with the implemented runtime skills.

    Stub skills are intentionally left unregistered; the resource registry
    is responsible for reporting which catalog entries are disabled.
    """
    sc = getattr(config, "SKILL_CONFIGS", {})
    registry = SkillRegistry()

    def _cfg(skill_name: str, extra: dict | None = None) -> dict:
        merged = dict(extra or {})
        merged.update(sc.get(skill_name, {}))
        return resolve_skill_config_paths(skill_name, merged)

    # ── DTI (8 implemented) ──────────────────────────────────────────
    registry.register(ChEMBLSkill(_cfg("ChEMBL")))
    registry.register(BindingDBSkill(_cfg("BindingDB")))
    registry.register(DGIdbSkill(_cfg("DGIdb")))
    registry.register(OpenTargetsSkill(_cfg("Open Targets Platform")))
    registry.register(TTDSkill(_cfg("TTD")))
    registry.register(STITCHSkill(_cfg("STITCH")))
    registry.register(TarKGSkill(_cfg("TarKG")))
    registry.register(GDKDSkill(_cfg("GDKD")))
    registry.register(MolecularTargetsSkill(_cfg("Molecular Targets")))
    registry.register(MolecularTargetsDataSkill(_cfg("Molecular Targets Data")))

    # ── ADR (4 implemented) ──────────────────────────────────────────
    registry.register(FAERSSkill(_cfg("FAERS")))
    registry.register(SIDERSkill(_cfg("SIDER")))
    registry.register(nSIDESSkill(_cfg("nSIDES")))
    registry.register(ADReCSSkill(_cfg("ADReCS")))

    # ── Drug Knowledgebase (8 implemented) ───────────────────────────
    registry.register(UniD3Skill({
        "graphml_paths": getattr(config, "KG_ENDPOINTS", {}).get("unid3", {}),
        **_cfg("UniD3"),
    }))
    registry.register(DrugBankSkill(_cfg("DrugBank")))
    registry.register(IUPHARSkill(_cfg("IUPHAR/BPS Guide to Pharmacology")))
    registry.register(DrugCentralSkill(_cfg("DrugCentral")))
    registry.register(CPICSkill(_cfg("CPIC")))
    registry.register(PharmKGSkill(_cfg("PharmKG")))
    registry.register(WHOEssentialMedicinesSkill(_cfg("WHO Essential Medicines List")))
    registry.register(FDAOrangeBookSkill(_cfg("FDA Orange Book")))

    # ── Drug Mechanism (1 implemented) ───────────────────────────────
    registry.register(DrugMechDBSkill(_cfg("DRUGMECHDB")))

    # ── Drug Labeling (3 implemented) ────────────────────────────────
    registry.register(OpenFDASkill(_cfg("openFDA Human Drug")))
    registry.register(DailyMedSkill(_cfg("DailyMed")))
    registry.register(MedlinePlusSkill(_cfg("MedlinePlus Drug Info")))

    # ── Drug Ontology (4 implemented) ────────────────────────────────
    registry.register(RxNormSkill(_cfg("RxNorm")))
    registry.register(ChEBISkill(_cfg("ChEBI")))
    registry.register(ATCSkill(_cfg("ATC/DDD")))
    registry.register(NDFRTSkill(_cfg("NDF-RT")))

    # ── Drug Repurposing (6 implemented) ─────────────────────────────
    registry.register(RepoDBSkill(_cfg("RepoDB")))
    registry.register(DRKGSkill(_cfg("DRKG")))
    registry.register(OREGANOSkill(_cfg("OREGANO")))
    registry.register(RepurposingHubSkill(_cfg("Drug Repurposing Hub")))
    registry.register(DrugRepoBankSkill(_cfg("DrugRepoBank")))
    registry.register(RepurposeDrugsSkill(_cfg("RepurposeDrugs")))

    # ── Pharmacogenomics (1 implemented) ─────────────────────────────
    registry.register(PharmGKBSkill(_cfg("PharmGKB")))

    # ── DDI (3 implemented) ──────────────────────────────────────────
    registry.register(MecDDISkill(_cfg("MecDDI")))
    registry.register(DDInterSkill(_cfg("DDInter")))
    registry.register(KEGGDrugSkill(_cfg("KEGG Drug")))

    # ── Drug Review (2 implemented) ──────────────────────────────────
    registry.register(WebMDReviewsSkill(_cfg("WebMD Drug Reviews")))
    registry.register(DrugsComReviewsSkill(_cfg("Drug Reviews (Drugs.com)")))

    # ── Drug Toxicity (4 implemented) ────────────────────────────────
    registry.register(UniToxSkill(_cfg("UniTox")))
    registry.register(LiverToxSkill(_cfg("LiverTox")))
    registry.register(DILIrankSkill(_cfg("DILIrank")))
    registry.register(DILISkill(_cfg("DILI")))

    # ── Drug Combination (2 implemented) ─────────────────────────────
    registry.register(DrugCombDBSkill(_cfg("DrugCombDB")))
    registry.register(DrugCombSkill(_cfg("DrugComb")))

    # ── Drug Molecular Property (1 implemented) ──────────────────────
    registry.register(GDSCSkill(_cfg("GDSC")))

    # ── Drug Disease (1 implemented) ─────────────────────────────────
    registry.register(SemaTyPSkill(_cfg("SemaTyP")))

    # ── Drug NLP (7 implemented) ─────────────────────────────────────
    registry.register(DDICorpusSkill(_cfg("DDI Corpus 2013")))
    registry.register(DrugProtSkill(_cfg("DrugProt")))
    registry.register(ADECorpusSkill(_cfg("ADE Corpus")))
    registry.register(CADECSkill(_cfg("CADEC")))
    registry.register(PsyTARSkill(_cfg("PsyTAR")))
    registry.register(TAC2017ADRSkill(_cfg("TAC 2017 ADR")))
    registry.register(PHEESkill(_cfg("PHEE")))

    # ── Web Search (always-on) ───────────────────────────────────────
    registry.register(WebSearchSkill(_cfg("WebSearch")))

    return registry
