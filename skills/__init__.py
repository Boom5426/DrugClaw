"""
DrugClaw RAG skills package.

The runtime resource registry is the authoritative source for active resource
counts and status. This package keeps the concrete skill implementations,
their catalog metadata, and the build_default_registry() factory.
"""
from .base import RAGSkill, DatasetRAGSkill, RetrievalResult, AccessMode
from .registry import SkillRegistry
from .skill_tree import SkillTree, SkillNode, Subcategory

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

    # ── DTI (8 implemented) ──────────────────────────────────────────
    registry.register(ChEMBLSkill(sc.get("ChEMBL", {})))
    registry.register(BindingDBSkill(sc.get("BindingDB", {})))
    registry.register(DGIdbSkill(sc.get("DGIdb", {})))
    registry.register(OpenTargetsSkill(sc.get("Open Targets Platform", {})))
    registry.register(TTDSkill(sc.get("TTD", {})))
    registry.register(STITCHSkill(sc.get("STITCH", {})))
    registry.register(TarKGSkill(sc.get("TarKG", {})))
    registry.register(GDKDSkill(sc.get("GDKD", {})))
    registry.register(MolecularTargetsSkill(sc.get("Molecular Targets", {})))
    registry.register(MolecularTargetsDataSkill(sc.get("Molecular Targets Data", {})))

    # ── ADR (4 implemented) ──────────────────────────────────────────
    registry.register(FAERSSkill(sc.get("FAERS", {})))
    registry.register(SIDERSkill(sc.get("SIDER", {})))
    registry.register(nSIDESSkill(sc.get("nSIDES", {})))
    registry.register(ADReCSSkill(sc.get("ADReCS", {})))

    # ── Drug Knowledgebase (8 implemented) ───────────────────────────
    registry.register(UniD3Skill({
        "graphml_paths": getattr(config, "KG_ENDPOINTS", {}).get("unid3", {}),
        **sc.get("UniD3", {}),
    }))
    registry.register(DrugBankSkill(sc.get("DrugBank", {})))
    registry.register(IUPHARSkill(sc.get("IUPHAR/BPS Guide to Pharmacology", {})))
    registry.register(DrugCentralSkill(sc.get("DrugCentral", {})))
    registry.register(CPICSkill(sc.get("CPIC", {})))
    registry.register(PharmKGSkill(sc.get("PharmKG", {})))
    registry.register(WHOEssentialMedicinesSkill(sc.get("WHO Essential Medicines List", {})))
    registry.register(FDAOrangeBookSkill(sc.get("FDA Orange Book", {})))

    # ── Drug Mechanism (1 implemented) ───────────────────────────────
    registry.register(DrugMechDBSkill(sc.get("DRUGMECHDB", {})))

    # ── Drug Labeling (3 implemented) ────────────────────────────────
    registry.register(OpenFDASkill(sc.get("openFDA Human Drug", {})))
    registry.register(DailyMedSkill(sc.get("DailyMed", {})))
    registry.register(MedlinePlusSkill(sc.get("MedlinePlus Drug Info", {})))

    # ── Drug Ontology (4 implemented) ────────────────────────────────
    registry.register(RxNormSkill(sc.get("RxNorm", {})))
    registry.register(ChEBISkill(sc.get("ChEBI", {})))
    registry.register(ATCSkill(sc.get("ATC/DDD", {})))
    registry.register(NDFRTSkill(sc.get("NDF-RT", {})))

    # ── Drug Repurposing (6 implemented) ─────────────────────────────
    registry.register(RepoDBSkill(sc.get("RepoDB", {})))
    registry.register(DRKGSkill(sc.get("DRKG", {})))
    registry.register(OREGANOSkill(sc.get("OREGANO", {})))
    registry.register(RepurposingHubSkill(sc.get("Drug Repurposing Hub", {})))
    registry.register(DrugRepoBankSkill(sc.get("DrugRepoBank", {})))
    registry.register(RepurposeDrugsSkill(sc.get("RepurposeDrugs", {})))

    # ── Pharmacogenomics (1 implemented) ─────────────────────────────
    registry.register(PharmGKBSkill(sc.get("PharmGKB", {})))

    # ── DDI (3 implemented) ──────────────────────────────────────────
    registry.register(MecDDISkill(sc.get("MecDDI", {})))
    registry.register(DDInterSkill(sc.get("DDInter", {})))
    registry.register(KEGGDrugSkill(sc.get("KEGG Drug", {})))

    # ── Drug Review (2 implemented) ──────────────────────────────────
    registry.register(WebMDReviewsSkill(sc.get("WebMD Drug Reviews", {})))
    registry.register(DrugsComReviewsSkill(sc.get("Drug Reviews (Drugs.com)", {})))

    # ── Drug Toxicity (4 implemented) ────────────────────────────────
    registry.register(UniToxSkill(sc.get("UniTox", {})))
    registry.register(LiverToxSkill(sc.get("LiverTox", {})))
    registry.register(DILIrankSkill(sc.get("DILIrank", {})))
    registry.register(DILISkill(sc.get("DILI", {})))

    # ── Drug Combination (2 implemented) ─────────────────────────────
    registry.register(DrugCombDBSkill(sc.get("DrugCombDB", {})))
    registry.register(DrugCombSkill(sc.get("DrugComb", {})))

    # ── Drug Molecular Property (1 implemented) ──────────────────────
    registry.register(GDSCSkill(sc.get("GDSC", {})))

    # ── Drug Disease (1 implemented) ─────────────────────────────────
    registry.register(SemaTyPSkill(sc.get("SemaTyP", {})))

    # ── Drug NLP (7 implemented) ─────────────────────────────────────
    registry.register(DDICorpusSkill(sc.get("DDI Corpus 2013", {})))
    registry.register(DrugProtSkill(sc.get("DrugProt", {})))
    registry.register(ADECorpusSkill(sc.get("ADE Corpus", {})))
    registry.register(CADECSkill(sc.get("CADEC", {})))
    registry.register(PsyTARSkill(sc.get("PsyTAR", {})))
    registry.register(TAC2017ADRSkill(sc.get("TAC 2017 ADR", {})))
    registry.register(PHEESkill(sc.get("PHEE", {})))

    # ── Web Search (always-on) ───────────────────────────────────────
    registry.register(WebSearchSkill(sc.get("WebSearch", {})))

    return registry
