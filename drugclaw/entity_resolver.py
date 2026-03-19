"""
Entity Resolver — fuzzy matching and variant expansion for extracted entities.

Handles two scenarios:
  1. LOCAL_FILE skills: fuzzy-match user entities against a prebuilt index of
     known entity names using ``difflib.get_close_matches()`` (zero extra deps).
  2. REST_API / CLI skills: ask the LLM to generate canonical names, common
     synonyms, and likely typo corrections so that API queries cast a wider net.

The resolver sits between entity extraction (PlannerAgent / RetrieverAgent)
and the Code Agent, expanding the entity dict with high-confidence variants
so that downstream retrieval is more robust to case differences and typos.
"""
from __future__ import annotations

import csv
import difflib
import logging
import os
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

# Entity types we attempt to resolve
_RESOLVABLE_TYPES = {"drug", "gene", "disease", "pathway"}

# difflib similarity threshold (0–1); 0.6 is the sweet-spot for biomedical names
_DEFAULT_CUTOFF = 0.6

# Maximum number of fuzzy matches to return per entity
_MAX_FUZZY_MATCHES = 3

# Maximum number of LLM variant suggestions per entity
_MAX_LLM_VARIANTS = 4


class EntityResolver:
    """Expand and normalise extracted entities before retrieval."""

    def __init__(
        self,
        skill_registry=None,
        llm_client=None,
    ):
        self._skill_registry = skill_registry
        self._llm = llm_client
        # Lazily-built index:  entity_type -> set of known lowercase names
        self._local_index: Optional[Dict[str, Set[str]]] = None
        # Original-case mapping: lowercase -> original form (first seen)
        self._case_map: Dict[str, str] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def resolve(
        self,
        entities: Dict[str, List[str]],
        skill_names: List[str],
        *,
        use_llm: bool = True,
    ) -> Dict[str, List[str]]:
        """
        Return an expanded copy of *entities* with fuzzy / variant matches.

        Parameters
        ----------
        entities : dict
            Canonical entity dict, e.g. ``{"drug": ["imatinib"]}``.
        skill_names : list[str]
            Skills that will be queried (used to decide local vs API strategy).
        use_llm : bool
            Whether to call the LLM for variant generation (disable in tests).

        Returns
        -------
        dict with the same keys but potentially expanded value lists.
        Original entities always come first in each list.
        """
        if not entities:
            return entities

        has_local = self._has_local_skills(skill_names)
        has_api = self._has_api_skills(skill_names)

        expanded: Dict[str, List[str]] = {}

        for etype, enames in entities.items():
            if etype not in _RESOLVABLE_TYPES or not enames:
                expanded[etype] = list(enames)
                continue

            seen: Set[str] = set()
            result: List[str] = []

            # Always keep originals first
            for name in enames:
                low = name.strip().lower()
                if low and low not in seen:
                    result.append(name.strip())
                    seen.add(low)

            # Strategy 1: fuzzy match against local index
            if has_local:
                for name in list(enames):
                    for match in self._fuzzy_match_local(name, etype):
                        low = match.lower()
                        if low not in seen:
                            result.append(match)
                            seen.add(low)

            # Strategy 2: LLM variant expansion (for API skills or when local had no matches)
            if has_api and use_llm and self._llm is not None:
                variants = self._generate_variants_via_llm(enames, etype)
                for v in variants:
                    low = v.strip().lower()
                    if low and low not in seen:
                        result.append(v.strip())
                        seen.add(low)

            expanded[etype] = result

        if expanded != entities:
            logger.info("Entity resolver expanded: %s -> %s", entities, expanded)

        return expanded

    # ------------------------------------------------------------------
    # Local index: build + fuzzy match
    # ------------------------------------------------------------------

    def build_local_index(self) -> None:
        """
        Scan all LOCAL_FILE skills and build an in-memory index of known
        entity names.  Called lazily on first fuzzy match attempt.
        """
        if self._skill_registry is None:
            self._local_index = {}
            return

        index: Dict[str, Set[str]] = {t: set() for t in _RESOLVABLE_TYPES}

        local_skills = self._skill_registry.list_by_access_mode("LOCAL_FILE")
        for skill_name in local_skills:
            skill = self._skill_registry.get_skill(skill_name)
            if skill is None:
                continue
            try:
                names = self._extract_entity_names_from_skill(skill)
                for etype, name_set in names.items():
                    if etype in index:
                        index[etype].update(name_set)
            except Exception as exc:
                logger.debug(
                    "Could not index skill %s: %s", skill_name, exc,
                )

        # Build case map (lowercase -> original)
        for etype, name_set in index.items():
            for name in name_set:
                low = name.lower()
                if low not in self._case_map:
                    self._case_map[low] = name

        total = sum(len(s) for s in index.values())
        logger.info(
            "Entity resolver indexed %d names from %d LOCAL_FILE skills",
            total, len(local_skills),
        )
        self._local_index = index

    def _ensure_index(self) -> Dict[str, Set[str]]:
        if self._local_index is None:
            self.build_local_index()
        return self._local_index  # type: ignore[return-value]

    def _fuzzy_match_local(
        self,
        entity_name: str,
        entity_type: str,
        cutoff: float = _DEFAULT_CUTOFF,
    ) -> List[str]:
        """
        Find close matches for *entity_name* in the local index.

        Returns original-case versions of matched names.
        """
        index = self._ensure_index()
        candidates = index.get(entity_type, set())
        if not candidates:
            return []

        # Build lowercase candidate list for matching
        lower_candidates = list({c.lower() for c in candidates})
        query_lower = entity_name.strip().lower()

        matches = difflib.get_close_matches(
            query_lower,
            lower_candidates,
            n=_MAX_FUZZY_MATCHES,
            cutoff=cutoff,
        )

        # Map back to original case
        result = []
        for m in matches:
            original = self._case_map.get(m, m)
            if original.lower() != query_lower:  # skip exact matches
                result.append(original)
        return result

    @staticmethod
    def _extract_entity_names_from_skill(skill) -> Dict[str, Set[str]]:
        """
        Extract known entity names from a LOCAL_FILE skill's loaded data.

        Handles the common patterns:
          - skill._rows / skill._data (list of dicts)
          - skill._drug_index / skill._gene_index (pre-built dicts)
        """
        names: Dict[str, Set[str]] = {t: set() for t in _RESOLVABLE_TYPES}

        # Pattern 1: Pre-built index dicts (most LOCAL_FILE skills)
        for attr, etype in (
            ("_drug_index", "drug"),
            ("_gene_index", "gene"),
            ("_target_index", "gene"),
            ("_disease_index", "disease"),
            ("_pathway_index", "pathway"),
        ):
            idx = getattr(skill, attr, None)
            if isinstance(idx, dict):
                # Index keys are usually lowercase; values are lists of rows
                names[etype].update(str(k) for k in idx.keys() if k)

        # Pattern 2: Raw row data
        rows = getattr(skill, "_rows", None) or getattr(skill, "_data", None)
        if isinstance(rows, list) and rows:
            # Common drug columns
            drug_cols = {
                "drug", "drug_name", "compound", "compound_name",
                "pert_iname", "DrugName", "Drug",
                "drug1", "drug2", "Drug1", "Drug2",
                "drug_row", "drug_col",
            }
            gene_cols = {
                "gene", "gene_name", "target", "target_name",
                "Gene", "GENE", "Target", "protein",
            }
            disease_cols = {
                "disease", "disease_name", "indication",
                "Disease", "Indication", "DiseaseName",
                "disease_area", "CancerType", "cancer_type",
            }

            sample = rows[:5000]  # cap for speed
            for row in sample:
                if not isinstance(row, dict):
                    continue
                for col in drug_cols:
                    val = row.get(col)
                    if val and isinstance(val, str) and len(val) > 1:
                        names["drug"].add(val.strip())
                for col in gene_cols:
                    val = row.get(col)
                    if val and isinstance(val, str) and len(val) > 1:
                        names["gene"].add(val.strip())
                for col in disease_cols:
                    val = row.get(col)
                    if val and isinstance(val, str) and len(val) > 1:
                        names["disease"].add(val.strip())

        return names

    # ------------------------------------------------------------------
    # LLM variant generation (for API-backed skills)
    # ------------------------------------------------------------------

    def _generate_variants_via_llm(
        self,
        entity_names: List[str],
        entity_type: str,
    ) -> List[str]:
        """
        Ask the LLM to produce canonical names, common synonyms, and likely
        typo corrections for the given entities.

        Returns a flat list of unique variant strings (excluding originals).
        """
        if not self._llm or not entity_names:
            return []

        names_str = ", ".join(entity_names)
        prompt = f"""Given the following {entity_type} name(s) from a user query, generate variant forms that would help find them in biomedical databases.

Entity names: {names_str}
Entity type: {entity_type}

For each entity, provide:
1. The canonical/standardized name (e.g., INN name for drugs)
2. Common synonyms or trade names
3. If the name looks like it might contain a typo, suggest the corrected spelling

Return JSON only:
{{
  "variants": ["variant1", "variant2", ...]
}}

Rules:
- Only include high-confidence variants (do not hallucinate obscure names)
- Keep the list short (max {_MAX_LLM_VARIANTS} total variants across all entities)
- Do NOT repeat the original input names
- For drugs: include both generic and brand names if well-known
- For genes: include common aliases (e.g., TP53 / p53 / tumor protein p53)"""

        try:
            result = self._llm.generate_json(
                [{"role": "user", "content": prompt}],
                temperature=0.1,
            )
            variants = result.get("variants", [])
            if not isinstance(variants, list):
                return []

            # Deduplicate against originals
            original_lower = {n.strip().lower() for n in entity_names}
            cleaned = []
            for v in variants[:_MAX_LLM_VARIANTS]:
                v_str = str(v).strip()
                if v_str and v_str.lower() not in original_lower:
                    cleaned.append(v_str)
            return cleaned
        except Exception as exc:
            logger.debug("LLM variant generation failed: %s", exc)
            return []

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _has_local_skills(self, skill_names: List[str]) -> bool:
        if not self._skill_registry:
            return False
        for name in skill_names:
            skill = self._skill_registry.get_skill(name)
            if skill is not None and getattr(skill, "access_mode", "") == "LOCAL_FILE":
                return True
        return False

    def _has_api_skills(self, skill_names: List[str]) -> bool:
        if not self._skill_registry:
            return False
        for name in skill_names:
            skill = self._skill_registry.get_skill(name)
            if skill is not None and getattr(skill, "access_mode", "") in (
                "REST_API", "CLI",
            ):
                return True
        return False
