"""
Unified schema and abstract base classes for all DrugClaw RAG skills.

Every skill ingests a dict of entities + a free-text query and returns a list
of RetrievalResult objects.  The schema maps directly to the fields that
agent_retriever._build_subgraph() expects.

Skills are now organized by *subcategory* (matching the historical resource
inventory subcategory mapping) rather than by resource type (KG / Database /
Dataset).

Subcategories
-------------
adr               Adverse Drug Reaction (ADR)
drug_combination  Drug Combination/Synergy
drug_knowledgebase Drug Knowledgebase
drug_labeling     Drug Labeling/Info
drug_mechanism    Drug Mechanism
drug_molecular_property Drug Molecular Property
drug_nlp          Drug NLP/Text Mining
drug_ontology     Drug Ontology/Terminology
drug_repurposing  Drug Repurposing
drug_review       Drug Review/Patient Report
drug_toxicity     Drug Toxicity
ddi               Drug-Drug Interaction (DDI)
dti               Drug-Target Interaction (DTI)
drug_disease      Drug–Disease Associations
pharmacogenomics  Pharmacogenomics

Access modes
------------
REST_API   HTTP REST / GraphQL endpoint (no install required)
CLI        Python package with CLI interface (preferred when available)
LOCAL_FILE Pre-downloaded local file (CSV / TSV / JSON / GraphML)
DATASET    Labelled benchmark dataset (mask during evaluation)
"""
from __future__ import annotations

import logging
import os
import re
import shutil
import subprocess
import time
import unicodedata
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from difflib import get_close_matches
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Access-mode constants
# ---------------------------------------------------------------------------

class AccessMode:
    REST_API   = "REST_API"
    CLI        = "CLI"
    LOCAL_FILE = "LOCAL_FILE"
    DATASET    = "DATASET"


# ---------------------------------------------------------------------------
# Unified output schema
# ---------------------------------------------------------------------------

@dataclass
class RetrievalResult:
    """
    Canonical output record produced by every RAG skill.

    Fields used by agent_retriever._build_subgraph
    -----------------------------------------------
    source_entity  : str   – name of the source node
    source_type    : str   – entity type  (drug / gene / disease / protein / pathway / …)
    target_entity  : str   – name of the target node
    target_type    : str   – entity type of the target
    relationship   : str   – edge label (e.g. "treats", "targets", "inhibits")
    weight         : float – always 1.0; relationship exists or it does not
    source         : str   – skill/database name (used as provenance label)

    Extra context (optional, carried through to EvidenceSubgraph.metadata)
    -----------------------------------------------------------------------
    evidence_text  : free-text explanation, abstract snippet, mechanism note …
    sources        : list of provenance references — any mix of:
                       • PubMed IDs  ("PMID:12345678")
                       • URLs        ("https://www.ebi.ac.uk/chembl/…")
                       • Paper titles ("Drug Repurposing … Nature 2021")
                     Not every resource provides citations; leave empty if none.
    skill_category : subcategory string (e.g. "dti", "adr", "drug_repurposing")
    metadata       : any additional key-value pairs
    """

    # --- required core fields (mirrors _build_subgraph contract) ---
    source_entity: str
    source_type: str
    target_entity: str
    target_type: str
    relationship: str
    weight: float        # always 1.0
    source: str          # skill name used as DB provenance label

    # --- optional enrichment ---
    evidence_text: Optional[str] = None
    sources: List[str] = field(default_factory=list)   # PMIDs / URLs / titles
    skill_category: str = "unknown"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialise to a plain dict (compatible with existing pipeline)."""
        d = {
            "source_entity": self.source_entity,
            "source_type": self.source_type,
            "target_entity": self.target_entity,
            "target_type": self.target_type,
            "relationship": self.relationship,
            "weight": self.weight,
            "source": self.source,
            "skill_category": self.skill_category,
        }
        if self.evidence_text is not None:
            d["evidence_text"] = self.evidence_text
        if self.sources:
            d["sources"] = self.sources
        if self.metadata:
            d.update(self.metadata)
        return d


# ---------------------------------------------------------------------------
# Abstract base skill
# ---------------------------------------------------------------------------

class RAGSkill(ABC):
    """
    Abstract base class that every RAG skill must implement.

    Subclasses must define:
      - name          : unique string identifier (e.g. "ChEMBL", "DGIdb")
      - subcategory   : one of the 15 subcategory keys (e.g. "dti", "adr")
      - resource_type : 'KG' | 'Database' | 'Dataset'  (kept for compat)
      - access_mode   : AccessMode constant
      - aim           : short purpose string from the survey
      - data_range    : coverage description
      - retrieve()    : the actual retrieval logic
    """

    name: str = "UnnamedSkill"
    subcategory: str = "unknown"          # 15-category system
    resource_type: str = "unknown"        # KG / Database / Dataset  (legacy)
    access_mode: str = AccessMode.REST_API
    aim: str = ""
    data_range: str = ""
    _implemented: bool = False            # True only for skills with working example code

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config: Dict[str, Any] = config or {}

    @abstractmethod
    def retrieve(
        self,
        entities: Dict[str, List[str]],
        query: str = "",
        max_results: int = 50,
        **kwargs: Any,
    ) -> List[RetrievalResult]:
        """
        Main retrieval entry-point.

        Parameters
        ----------
        entities    : dict mapping entity_type -> list[entity_name]
                      e.g. {"drug": ["aspirin"], "gene": ["TP53"]}
        query       : optional free-text query for semantic retrieval
        max_results : upper bound on number of results to return

        Returns
        -------
        List of RetrievalResult objects with weight=1.0 for each found triplet.
        """

    def is_available(self) -> bool:
        """Return True if the skill can be used right now.

        Only implemented skills (with working example code + SKILL.md) are available.
        """
        return self._implemented

    def get_example_path(self) -> Optional[str]:
        """Return path to the example.py in this skill's directory, or None."""
        import inspect
        skill_dir = os.path.dirname(inspect.getfile(type(self)))
        example_path = os.path.join(skill_dir, "example.py")
        if os.path.exists(example_path):
            return example_path
        return None

    def get_skill_md_path(self) -> Optional[str]:
        """Return path to the SKILL.md in this skill's directory, or None."""
        import inspect
        skill_dir = os.path.dirname(inspect.getfile(type(self)))
        md_path = os.path.join(skill_dir, "SKILL.md")
        if os.path.exists(md_path):
            return md_path
        return None

    def get_example_code(self) -> str:
        """Read and return the example.py code for this skill."""
        path = self.get_example_path()
        if path:
            with open(path, "r", encoding="utf-8") as f:
                return f.read()
        return f"# No example code available for {self.name}"

    def get_skill_md(self) -> str:
        """Read and return the SKILL.md description for this skill."""
        path = self.get_skill_md_path()
        if path:
            with open(path, "r", encoding="utf-8") as f:
                return f.read()
        return f"No skill description available for {self.name}"

    def get_description(self) -> str:
        """Human-readable description used in LLM prompts."""
        return (
            f"{self.name} [{self.subcategory}] — aim: {self.aim} — "
            f"covers: {self.data_range}"
        )

    def planner_local_data_ready(self) -> Optional[bool]:
        """
        Best-effort readiness heuristic for LOCAL_FILE / DATASET skills.

        Returns:
          True  -> local data path is explicitly configured and exists
          False -> local/data path keys exist but target files are missing
          None  -> readiness cannot be inferred from config alone
        """
        if self.access_mode not in (AccessMode.LOCAL_FILE, AccessMode.DATASET):
            return True

        candidate_paths: List[str] = []
        candidate_dirs: List[str] = []
        for key, value in self.config.items():
            if not isinstance(value, str) or not value.strip():
                continue
            key_lower = key.lower()
            if key_lower.endswith(("_path", "_csv", "_tsv", "_json", "_xml", "_graphml")):
                candidate_paths.append(value)
            elif key_lower.endswith(("_dir", "_folder")):
                candidate_dirs.append(value)

        if candidate_paths or candidate_dirs:
            for raw in candidate_paths:
                if Path(raw).expanduser().exists():
                    return True
            for raw in candidate_dirs:
                p = Path(raw).expanduser()
                if p.exists() and p.is_dir():
                    return True
            return False

        return None

    def planner_profile(self) -> Dict[str, Any]:
        """Compact planning metadata consumed by the constrained retriever."""
        local_ready = self.planner_local_data_ready()
        evidence_type = {
            "KG": "knowledge_graph",
            "Database": "curated_database",
            "Dataset": "dataset",
            "API": "api",
            "WebSearch": "web_search",
        }.get(self.resource_type, self.resource_type.lower())

        tags = {
            self.subcategory,
            self.resource_type.lower(),
            self.access_mode.lower(),
        }
        if self.access_mode == AccessMode.LOCAL_FILE:
            tags.add("offline_ready" if local_ready else "offline_needed")
        if self.access_mode == AccessMode.DATASET:
            tags.add("benchmark_dataset")
        if self.access_mode == AccessMode.CLI:
            tags.add("cli_preferred")
        if self.access_mode == AccessMode.REST_API:
            tags.add("live_api")

        package_flags = self._planner_package_flags()
        for tag_name, enabled in package_flags.items():
            if enabled:
                tags.add(tag_name)

        return {
            "name": self.name,
            "subcategory": self.subcategory,
            "resource_type": self.resource_type,
            "access_mode": self.access_mode,
            "aim": self.aim,
            "data_range": self.data_range,
            "available": bool(self.is_available()),
            "implemented": bool(self._implemented),
            "local_data_ready": local_ready,
            "evidence_type": evidence_type,
            **package_flags,
            "tags": sorted(tags),
        }

    def _planner_package_flags(self) -> Dict[str, bool]:
        snapshot = getattr(self, "resource_package_snapshot", None)
        if snapshot is None and isinstance(self.config, dict):
            snapshot = self.config.get("resource_package")

        def _read_flag(flag_name: str) -> bool:
            if snapshot is None:
                return False
            if isinstance(snapshot, dict):
                return bool(snapshot.get(flag_name, False))
            return bool(getattr(snapshot, flag_name, False))

        return {
            "has_dataset_bundle": _read_flag("has_dataset_bundle"),
            "has_protocol": _read_flag("has_protocol"),
            "has_how_to": _read_flag("has_how_to"),
            "has_knowhow": _read_flag("has_knowhow"),
            "has_software_dependency": _read_flag("has_software_dependency"),
        }

    # ---- helpers ----------------------------------------------------------

    @staticmethod
    def _clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
        return max(lo, min(hi, value))

    def _retry(self, fn, retries: int = 3, delay: float = 1.0):
        """Simple retry wrapper for network calls."""
        last_exc: Optional[Exception] = None
        for attempt in range(retries):
            try:
                return fn()
            except Exception as exc:
                last_exc = exc
                time.sleep(delay * (2 ** attempt))
        raise last_exc  # type: ignore[misc]

    # ---- fuzzy name matching helpers -------------------------------------

    @staticmethod
    def _norm(name: str) -> str:
        """
        Normalise a drug/entity name for fuzzy matching.

        Steps: Unicode NFKD decomposition → ASCII-fold → lowercase →
               collapse non-alphanumeric runs to single spaces → strip.

        Examples
        --------
        "Aspirin"            → "aspirin"
        "ACETAMINOPHEN"      → "acetaminophen"
        "metformin HCl"      → "metformin hcl"
        "5-fluorouracil"     → "5 fluorouracil"
        "Atorvastatin (10)"  → "atorvastatin 10"
        """
        n = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
        n = re.sub(r"[^a-z0-9]", " ", n.lower())
        return re.sub(r"\s+", " ", n).strip()

    def _build_fuzzy_index(
        self,
        keys: Iterable[str],
        attr: str = "_fuzzy_norm_map",
    ) -> None:
        """
        Build a normalised→canonical lookup dict for a set of index keys.

        Call once after loading data.  Pass the same keys that are used as
        keys in your internal index dict (e.g. ``self._drug_index.keys()``).
        The result is stored as an instance attribute (default
        ``_fuzzy_norm_map``) so that :meth:`_fuzzy_get` can use it.

        Parameters
        ----------
        keys : iterable of str
            Canonical key strings (as stored in the index dict).
        attr : str
            Name of the instance attribute to store the map in.
            Use different names when you have multiple index dicts
            (e.g. ``"_drug_fuzzy"`` and ``"_disease_fuzzy"``).
        """
        setattr(self, attr, {self._norm(k): k for k in keys})

    def _fuzzy_get(
        self,
        query: str,
        index: Dict[str, list],
        norm_map_attr: str = "_fuzzy_norm_map",
        cutoff: float = 0.85,
    ) -> list:
        """
        Fuzzy lookup in *index* for *query*.

        Returns a flat list of all values stored under the best-matching
        key(s) in *index*.  Empty list if nothing found.

        Matching strategy (in priority order):
        1. Exact lowercase match — ``index[query.lower()]``
        2. Normalised exact match — via pre-built norm→canonical map
        3. Substring match — norm(query) ⊆ norm(key) or vice-versa
        4. ``difflib.get_close_matches`` with *cutoff*

        Parameters
        ----------
        query       : raw drug name from user input (any case, may have typos)
        index       : dict mapping canonical_lower → List[…]
        norm_map_attr : instance attribute name of the pre-built norm map
                      (built by :meth:`_build_fuzzy_index`)
        cutoff      : similarity threshold for difflib (default 0.85)
        """
        # 1. Direct exact lowercase match (fast path, covers most cases)
        q_lower = query.lower()
        if q_lower in index:
            return list(index[q_lower])

        # Retrieve the pre-built norm map (may be absent if not built yet)
        norm_map: Dict[str, str] = getattr(self, norm_map_attr, {})

        # 2. Build lazily from index keys when not pre-built
        if not norm_map:
            norm_map = {self._norm(k): k for k in index.keys()}

        q_norm = self._norm(query)

        # 3. Normalised exact match
        if q_norm in norm_map:
            return list(index.get(norm_map[q_norm], []))

        # 4. Substring match (handles salts: "metformin hcl" ↔ "metformin")
        results: list = []
        for k_norm, canon in norm_map.items():
            if q_norm and (q_norm in k_norm or k_norm in q_norm):
                results.extend(index.get(canon, []))
        if results:
            return results

        # 5. difflib fuzzy match
        close = get_close_matches(q_norm, list(norm_map.keys()), n=3, cutoff=cutoff)
        for cn in close:
            results.extend(index.get(norm_map[cn], []))
        return results


# ---------------------------------------------------------------------------
# CLI-capable skill mixin
# ---------------------------------------------------------------------------

class CLISkillMixin:
    """
    Mixin for skills that can use a CLI tool (Python package with CLI interface)
    in preference to raw REST calls.

    Inspired by OpenClaw Medical Skills' approach of preferring installed
    CLI/Python-package tools over raw HTTP when available.

    Subclasses should:
      1. Set cli_package_name = "package_name"  (pip install name)
      2. Implement _cli_search(entities, query) -> List[RetrievalResult]
      3. Call self._try_cli_or_rest(entities, query) in retrieve()

    The mixin checks if the package is importable; if so, uses the CLI path;
    otherwise falls back to the REST path (_rest_search).
    """

    cli_package_name: str = ""
    _cli_checked: Optional[bool] = None

    def _cli_available(self) -> bool:
        if self._cli_checked is None:
            if not self.cli_package_name:
                self._cli_checked = False
            else:
                import importlib
                try:
                    importlib.import_module(self.cli_package_name.replace("-", "_"))
                    self._cli_checked = True
                    logger.debug(
                        "%s: CLI package '%s' available",
                        getattr(self, "name", "skill"),
                        self.cli_package_name,
                    )
                except ImportError:
                    self._cli_checked = False
                    logger.debug(
                        "%s: CLI package '%s' not found, using REST",
                        getattr(self, "name", "skill"),
                        self.cli_package_name,
                    )
        return self._cli_checked  # type: ignore[return-value]

    def _try_cli_or_rest(
        self,
        entities: Dict[str, List[str]],
        query: str = "",
        max_results: int = 50,
    ) -> List[RetrievalResult]:
        """Dispatch to CLI or REST depending on availability."""
        if self._cli_available():
            try:
                return self._cli_search(entities, query, max_results)  # type: ignore[attr-defined]
            except Exception as exc:
                logger.warning(
                    "%s: CLI search failed (%s), falling back to REST",
                    getattr(self, "name", "skill"),
                    exc,
                )
        return self._rest_search(entities, query, max_results)  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Dataset skill base class
# ---------------------------------------------------------------------------

class DatasetRAGSkill(RAGSkill):
    """
    Base class for benchmark dataset skills.

    Dataset skills are RAG sources like any other skill.  The key difference
    is that they contain *labelled* drug–disease (or drug–AE) pairs which can
    serve as both:

      1. Evidence during inference  — register the skill normally.
      2. Held-out evaluation pairs  — simply do NOT register this skill in
         the SkillRegistry when benchmarking on it.  The system must then
         reason purely from the other registered skills.

    ``get_all_pairs()`` enumerates every labelled record, which is useful for
    setting up evaluation loops (e.g. iterating over drug–disease pairs and
    calling the pipeline for each).
    """

    resource_type: str = "Dataset"
    access_mode: str = AccessMode.DATASET

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self._loaded = False

    @abstractmethod
    def get_all_pairs(self) -> List[Dict[str, Any]]:
        """
        Return every labelled pair in the dataset as a list of dicts.

        Each dict must have at minimum:
          drug     : str   drug common name
          disease  : str   disease / indication / adverse event name
          label    : str   outcome label (e.g. "Approved", "positive")
        """

    def _ensure_loaded(self) -> None:
        if not self._loaded:
            self._load()
            self._loaded = True

    def _load(self) -> None:
        """Override to populate data from files / network."""
