"""
Retriever Agent — selects skills and delegates querying to the Code Agent.

The Retriever Agent is now focused on:
  1. Analyzing the query and selecting the best skills (via LLM or resource_filter)
  2. Extracting key entities from the query
  3. Delegating actual querying to the Code Agent
  4. Collecting results as free-form text (no forced RetrievalResult schema)

The old _build_subgraph() is replaced by the Graph Build Agent (graph mode)
or results go directly as text to the Responder Agent (simple mode).
"""
import re
from typing import List, Dict, Any, Optional

from .models import AgentState
from .llm_client import LLMClient
from .query_plan import (
    QueryPlan,
    build_fallback_query_plan,
    is_direct_target_lookup,
    normalize_question_type,
    prioritize_target_lookup_skills,
)
from .skills.registry import SkillRegistry
from .agent_coder import CoderAgent
from .evidence import build_evidence_items_for_skill
from .entity_resolver import EntityResolver


class RetrieverAgent:
    """
    Agent responsible for selecting skills and orchestrating retrieval.

    Uses LLM to navigate the runtime skill tree, then hands off
    to the Code Agent for actual querying.  Results come back as text.
    """

    def __init__(
        self,
        llm_client: LLMClient,
        skill_registry: SkillRegistry,
        coder_agent: Optional[CoderAgent] = None,
        resource_registry=None,
        entity_resolver: Optional[EntityResolver] = None,
    ):
        self.llm = llm_client
        self.skill_registry = skill_registry
        self.coder = coder_agent or CoderAgent(llm_client, skill_registry)
        self.resource_registry = resource_registry
        self.entity_resolver = entity_resolver or EntityResolver(
            skill_registry=skill_registry,
            llm_client=llm_client,
        )

    # ------------------------------------------------------------------
    # Prompt builders
    # ------------------------------------------------------------------

    def get_system_prompt(self) -> str:
        """System prompt for skill selection."""
        tree_ctx = self.skill_registry.skill_tree_prompt
        return f"""You are the Retriever Agent of DrugClaw — a drug-specialized agentic RAG system grounded in the current runtime resource registry across 15 subcategories (DTI, ADR, DDI, drug mechanisms, pharmacogenomics, drug ontology, drug labeling, drug repurposing, drug toxicity, drug combinations, drug molecular properties, drug-disease associations, drug knowledge bases, drug reviews, and drug NLP datasets).

Your role is to:
1. Analyze the user's drug-related query
2. Navigate the 15-subcategory Skill Tree to select the most relevant resources
3. Extract key entities (drugs, genes, diseases, pathways) from the query
4. Decide which skills to query

---
{tree_ctx}
---

Each skill is marked:
  ✓ = implemented and available to call
  ○ = catalogued but not yet available (do NOT include in query_plan)

Skills also show their access mode:
  [CLI]        — Python package CLI (preferred; auto-detected if installed)
  [LOCAL_FILE] — requires pre-downloaded local data file under resources_metadata/
  [DATASET]    — benchmark dataset (mask when evaluating on it)
  (no tag)     — public REST/GraphQL API (always available)

For [LOCAL_FILE] skills, follow this policy:
- First prefer files already present in the local repository under resources_metadata/
- If a file is missing, prefer the curated mirror dataset `Mike2481/DrugClaw_resources_data`
- Do not assume the original upstream download URL is still alive unless the user explicitly asks to verify it
- If neither local nor mirrored data is available, avoid selecting that skill unless it is essential and you note the dependency clearly

Navigation strategy by subcategory:
- Drug targets, bioactivity, binding              → [dti]
- Side effects, pharmacovigilance                 → [adr]
- General drug info, encyclopedias, KGs           → [drug_knowledgebase]
- Mechanism of action paths                       → [drug_mechanism]
- FDA labels, prescribing info, dosing            → [drug_labeling]
- Drug codes, ATC, RxNorm, ontologies             → [drug_ontology]
- New indications, repositioning                  → [drug_repurposing]
- Variant-drug, PGx guidelines                   → [pharmacogenomics]
- Polypharmacy, drug interactions                 → [ddi]
- Hepatotoxicity, DILI, organ tox                → [drug_toxicity]
- Synergy, combination therapy                    → [drug_combination]
- QSAR, cell-line sensitivity, IC50              → [drug_molecular_property]
- Disease-drug semantic associations              → [drug_disease]
- Patient reviews, real-world effectiveness       → [drug_review]
- NLP corpora, relation extraction                → [drug_nlp]

Only select skills marked ✓ in your query_plan.
Respond with a structured plan for retrieval."""

    def get_query_planning_prompt(
        self,
        query: str,
        omics_constraints: str,
        iteration: int,
        suggested_skills: List[str],
    ) -> str:
        suggestion_str = (
            "Suggested skills (keyword-matched to your query, all ✓):\n  " +
            ", ".join(suggested_skills)
            if suggested_skills
            else "No keyword-matched suggestions — use the tree in the system prompt."
        )
        return f"""Query: {query}

Biological Constraints:
{omics_constraints}

{suggestion_str}

Current Iteration: {iteration}

Use the Skill Tree in the system prompt to navigate to the right domain, then
select the best ✓-marked skills for this query.

Provide your plan in JSON format:
{{
    "key_entities": {{
        "drugs": ["drug1", "drug2"],
        "genes": ["gene1", "gene2"],
        "diseases": ["disease1"],
        "pathways": ["pathway1"],
        "other": []
    }},
    "highly_related_entities": {{
        "drugs": ["related_drug1"],
        "genes": ["related_gene1"],
        "diseases": ["related_disease1"],
        "pathways": [],
        "other": []
    }},
    "selected_skills": ["ChEMBL", "DGIdb"],
    "reasoning": "Explanation of which skill-tree domain you used and why these skills were selected"
}}"""

    # ------------------------------------------------------------------
    # Main execute
    # ------------------------------------------------------------------

    def execute(self, state: AgentState) -> AgentState:
        """
        Execute retrieval step:
          1. Plan (LLM selects skills + extracts entities)
          2. Delegate to Code Agent for actual querying
          3. Store results as text in state.retrieved_text
        """
        print(f"\n[Retriever Agent] Iteration {state.iteration}")
        resource_filter = getattr(state, "resource_filter", [])
        if resource_filter:
            print(f"[Retriever Agent] resource_filter={resource_filter}")

        # Prepare omics constraints string
        omics_str = self._format_omics_constraints(state.omics_constraints)
        planner_output = getattr(state, "query_plan", None)

        if (
            isinstance(planner_output, QueryPlan)
            and resource_filter
            and (
                not planner_output.entities
                or normalize_question_type(planner_output.question_type) == "unknown"
            )
        ):
            planner_output = self._build_resource_filter_query_plan(
                state.original_query,
                key_entities=planner_output.entities,
                resource_filter=resource_filter,
            )
            state.query_plan = planner_output

        if isinstance(planner_output, QueryPlan):
            query_plan = self._query_plan_to_retrieval_plan(
                planner_output,
                query=state.original_query,
                resource_filter=resource_filter,
            )
        elif resource_filter:
            query_plan = self._get_query_plan_filtered(
                state.original_query, omics_str, state.iteration, resource_filter,
            )
            state.query_plan = self._build_resource_filter_query_plan(
                state.original_query,
                key_entities=query_plan.get("key_entities", {}),
                resource_filter=resource_filter,
            )
        else:
            query_plan = self._get_query_plan(
                state.original_query, omics_str, state.iteration,
            )

        # Extract info from the plan
        key_entities = query_plan.get("key_entities", {})
        selected_skills = query_plan.get("selected_skills", [])

        if not self._normalize_entities_for_coder(key_entities) and isinstance(getattr(state, "query_plan", None), QueryPlan):
            key_entities = dict(state.query_plan.entities)

        # Backward compat: also check query_plan list for skill names
        if not selected_skills:
            selected_skills = [
                step.get("database", "")
                for step in query_plan.get("query_plan", [])
                if step.get("database")
            ]

        print(f"[Retriever Agent] Selected skills: {selected_skills}")
        print(f"[Retriever Agent] Key entities: {key_entities}")

        # Normalize entities for Code Agent
        entities = self._normalize_entities_for_coder(key_entities)

        # Fuzzy-resolve entities: expand with close matches / LLM variants
        if entities:
            entities = self.entity_resolver.resolve(
                entities,
                skill_names=selected_skills,
            )
            print(f"[Retriever Agent] Resolved entities: {entities}")

        execution_strategy = self._select_execution_strategy(state)

        # Delegate to Code Agent
        coder_result = self.coder.generate_and_execute(
            skill_names=selected_skills,
            entities=entities,
            query=state.original_query,
            execution_strategy=execution_strategy,
        )

        # Build a combined context string with query + entities + results
        context_parts = [
            f"Query: {state.original_query}",
            f"Key Entities: {key_entities}",
            f"Skills Used: {selected_skills}",
            "",
            coder_result["text"],
        ]

        # Include history if available
        if state.reasoning_steps:
            last_step = state.reasoning_steps[-1]
            context_parts.insert(
                3,
                f"Previous iteration answer (preview): "
                f"{last_step.intermediate_answer[:300]}...",
            )

        retrieved_text = "\n".join(context_parts)

        # Update state
        state.current_query_entities = key_entities
        state.retrieved_text = retrieved_text
        state.evidence_items = self._build_evidence_items(
            coder_result, selected_skills, state.original_query,
        )
        state.retrieval_diagnostics = self._build_retrieval_diagnostics(
            coder_result, selected_skills,
        )
        state.code_agent_code = "\n---\n".join(
            f"# Skill: {name}\n{info.get('code', '')}"
            for name, info in coder_result.get("per_skill", {}).items()
        )

        # Also populate retrieved_content for backward compat (flat list)
        state.retrieved_content = self._evidence_items_to_retrieved_content(
            state.evidence_items
        ) or self._text_to_retrieved_content(coder_result, selected_skills)

        print(f"[Retriever Agent] Retrieved {len(retrieved_text)} chars of text")
        print(f"[Retriever Agent] {len(state.retrieved_content)} backward-compat records")
        print(f"[Retriever Agent] {len(state.evidence_items)} structured evidence items")

        return state

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _format_omics_constraints(self, constraints) -> str:
        if constraints is None:
            return "No specific biological constraints provided."
        parts = []
        if constraints.gene_sets:
            parts.append(f"Genes: {', '.join(constraints.gene_sets[:10])}")
        if constraints.pathway_sets:
            parts.append(f"Pathways: {', '.join(constraints.pathway_sets[:10])}")
        if constraints.disease_terms:
            parts.append(f"Diseases: {', '.join(constraints.disease_terms)}")
        if constraints.tissue_types:
            parts.append(f"Tissues: {', '.join(constraints.tissue_types)}")
        return "\n".join(parts) if parts else "No specific constraints."

    @staticmethod
    def _select_execution_strategy(state: AgentState) -> str:
        plan = getattr(state, "query_plan", None)
        thinking_mode = str(getattr(state, "thinking_mode", ""))
        original_query = str(getattr(state, "original_query", "")).strip().lower()
        entities = getattr(plan, "entities", {}) if plan is not None else {}
        has_drug_entity = bool(getattr(entities, "get", lambda *_: [])("drug"))
        question_type = normalize_question_type(getattr(plan, "question_type", ""))
        direct_question_type = (
            is_direct_target_lookup(query=original_query, question_type=question_type)
            or any(marker in question_type for marker in ("label", "retrieval"))
        )
        direct_query_shape = has_drug_entity and any(
            marker in original_query
            for marker in (
                "target",
                "targets",
                "label",
                "prescribing",
                "information",
            )
        )
        if (
            thinking_mode == "simple"
            and plan is not None
            and (direct_question_type or direct_query_shape)
            and not getattr(plan, "requires_graph_reasoning", False)
        ):
            return "direct_retrieve"
        return "auto"

    def _query_plan_to_retrieval_plan(
        self,
        plan: QueryPlan,
        query: str,
        *,
        resource_filter: List[str],
    ) -> Dict[str, Any]:
        if resource_filter:
            selected_skills = self._filter_available_skills(list(resource_filter))
        else:
            combined_skill_hints = list(plan.preferred_skills) + list(
                self.skill_registry.get_skills_for_query(query)
            )
            combined_skill_hints = self._apply_query_skill_policy(
                combined_skill_hints,
                query=query,
                question_type=plan.question_type,
            )
            selected_skills = self._filter_available_skills(combined_skill_hints)
            if selected_skills:
                selected_skills = selected_skills[:3]

        return {
            "key_entities": dict(plan.entities),
            "selected_skills": selected_skills,
            "reasoning": "; ".join(plan.notes),
        }

    @staticmethod
    def _apply_query_skill_policy(
        skill_names: List[str],
        *,
        query: str,
        question_type: str,
    ) -> List[str]:
        unique_names = list(dict.fromkeys(skill_names))
        if is_direct_target_lookup(query=query, question_type=question_type):
            narrowed = prioritize_target_lookup_skills(unique_names)
            if narrowed:
                return narrowed
        return unique_names

    def _filter_available_skills(self, skill_names: List[str]) -> List[str]:
        if not skill_names:
            return []

        prioritized = self._prioritize_skill_names(skill_names, ready_only=True)
        if prioritized:
            return prioritized

        skill_names = self._prioritize_skill_names(skill_names, ready_only=False)
        available: List[str] = []
        for skill_name in skill_names:
            try:
                skill = self.skill_registry.get_skill(skill_name)
            except Exception:
                skill = None

            if skill is None:
                continue

            try:
                is_available = skill.is_available() if hasattr(skill, "is_available") else True
            except Exception:
                is_available = False

            if is_available:
                available.append(skill_name)

        return available

    def _prioritize_skill_names(
        self,
        skill_names: List[str],
        *,
        ready_only: bool,
    ) -> List[str]:
        unique_names = list(dict.fromkeys(skill_names))
        if not unique_names or self.resource_registry is None:
            return unique_names if not ready_only else []

        if hasattr(self.resource_registry, "prioritize_resource_names"):
            ranked = self.resource_registry.prioritize_resource_names(
                unique_names,
                ready_only=ready_only,
            )
            if ranked:
                return ranked

        ranked = []
        for index, skill_name in enumerate(unique_names):
            entry = getattr(self.resource_registry, "get_resource", lambda _: None)(skill_name)
            if entry is None:
                continue
            status = getattr(entry, "status", "")
            access_mode = getattr(entry, "access_mode", "")
            if ready_only and status != "ready":
                continue
            ranked.append(
                (
                    self._status_priority(status),
                    self._access_priority(access_mode),
                    index,
                    skill_name,
                )
            )
        return [name for _, _, _, name in sorted(ranked)]

    @staticmethod
    def _status_priority(status: str) -> int:
        order = {
            "ready": 0,
            "degraded": 1,
            "missing_dependency": 2,
            "missing_metadata": 3,
            "disabled": 4,
        }
        return order.get(status, 99)

    @staticmethod
    def _access_priority(access_mode: str) -> int:
        order = {
            "REST_API": 0,
            "CLI": 0,
            "LOCAL_FILE": 1,
            "DATASET": 2,
        }
        return order.get(access_mode, 3)

    @staticmethod
    def _normalize_entities_for_coder(
        key_entities: Dict[str, Any],
    ) -> Dict[str, List[str]]:
        singular_map = {
            "drug": "drug",
            "drugs": "drug",
            "gene": "gene",
            "genes": "gene",
            "disease": "disease",
            "diseases": "disease",
            "pathway": "pathway",
            "pathways": "pathway",
            "other": "other",
        }
        entities: Dict[str, List[str]] = {}
        for raw_key, raw_vals in key_entities.items():
            canonical = singular_map.get(str(raw_key).lower())
            if canonical is None:
                continue

            vals = raw_vals
            if isinstance(vals, str):
                vals = [vals]
            if not isinstance(vals, list):
                continue

            cleaned = [str(value).strip() for value in vals if str(value).strip()]
            if cleaned:
                entities[canonical] = cleaned
        return entities

    def _get_query_plan(
        self, query: str, omics_str: str, iteration: int,
    ) -> Dict[str, Any]:
        """Use LLM to select skills and extract entities."""
        suggested = self.skill_registry.get_skills_for_query(query)

        messages = [
            {"role": "system", "content": self.get_system_prompt()},
            {"role": "user", "content": self.get_query_planning_prompt(
                query, omics_str, iteration, suggested_skills=suggested,
            )},
        ]
        try:
            return self.llm.generate_json(messages)
        except Exception as e:
            print(f"[Retriever Agent] Error generating plan: {e}")
            return {
                "key_entities": {"drugs": [], "genes": [], "diseases": [], "pathways": []},
                "selected_skills": suggested[:3],
                "reasoning": "Fallback plan due to LLM error",
            }

    def _get_query_plan_filtered(
        self, query: str, omics_str: str, iteration: int,
        resource_filter: List[str],
    ) -> Dict[str, Any]:
        """Build a plan forcing the specified skills (skip LLM skill selection)."""
        entity_prompt = f"""Extract drug/gene/disease/pathway entities from this query for retrieval.
Query: {query}
Biological Constraints: {omics_str}

Respond in JSON:
{{
  "drugs": ["drugA"],
  "genes": ["geneA"],
  "diseases": ["diseaseA"],
  "pathways": [],
  "other": []
}}"""
        try:
            entities = self.llm.generate_json([{"role": "user", "content": entity_prompt}])
        except Exception:
            entities = self._infer_entities_from_query(query)

        return {
            "key_entities": entities,
            "selected_skills": resource_filter,
            "reasoning": f"resource_filter active: using {resource_filter}",
        }

    def _build_resource_filter_query_plan(
        self,
        query: str,
        *,
        key_entities: Dict[str, Any],
        resource_filter: List[str],
    ) -> QueryPlan:
        fallback = build_fallback_query_plan(query)
        entities = self._normalize_entities_for_coder(key_entities)
        if not entities:
            entities = self._infer_entities_from_query(query)
        question_type = self._infer_question_type_from_query(query)
        return QueryPlan(
            question_type=question_type or fallback.question_type,
            entities=entities,
            subquestions=[query] if query else [],
            preferred_skills=list(resource_filter),
            preferred_evidence_types=[],
            requires_graph_reasoning=False,
            requires_prediction_sources=False,
            requires_web_fallback=False,
            answer_risk_level="high" if question_type == "labeling" else fallback.answer_risk_level,
            notes=[f"resource_filter active: using {resource_filter}"],
        )

    @staticmethod
    def _infer_question_type_from_query(query: str) -> str:
        lowered = str(query).strip().lower()
        if not lowered:
            return "unknown"
        if any(marker in lowered for marker in ("prescribing", "label", "safety", "warning", "contraindication")):
            return "labeling"
        if is_direct_target_lookup(query=lowered):
            return "target_lookup"
        return "unknown"

    @staticmethod
    def _infer_entities_from_query(query: str) -> Dict[str, List[str]]:
        lowered = str(query).strip().lower()
        if not lowered:
            return {}

        for pattern in (
            r"information\s+is\s+available\s+for\s+([a-z0-9\-]+)",
            r"available\s+for\s+([a-z0-9\-]+)",
            r"targets?\s+of\s+([a-z0-9\-]+)",
            r"does\s+([a-z0-9\-]+)\s+target",
            r"about\s+([a-z0-9\-]+)$",
        ):
            match = re.search(pattern, lowered)
            if match:
                return {"drug": [match.group(1)]}
        return {}

    @staticmethod
    def _text_to_retrieved_content(
        coder_result: Dict[str, Any],
        skill_names: List[str],
    ) -> List[Dict[str, Any]]:
        """
        Convert coder output to backward-compatible retrieved_content list.
        Each entry is a minimal dict with source + text.
        """
        records = []
        for skill_name in skill_names:
            info = coder_result.get("per_skill", {}).get(skill_name, {})
            output = info.get("output", "").strip()
            if output:
                records.append({
                    "source": skill_name,
                    "source_entity": "",
                    "source_type": "",
                    "target_entity": "",
                    "target_type": "",
                    "relationship": "",
                    "weight": 1.0,
                    "evidence_text": output[:2000],
                    "skill_category": "",
                })
        return records

    def _build_evidence_items(
        self,
        coder_result: Dict[str, Any],
        skill_names: List[str],
        query: str,
    ) -> List[Dict[str, Any]]:
        items = []
        for skill_name in skill_names:
            skill = self.skill_registry.get_skill(skill_name)
            info = coder_result.get("per_skill", {}).get(skill_name, {})
            records = info.get("records", []) or []
            items.extend(
                build_evidence_items_for_skill(
                    skill_name=skill_name,
                    records=records,
                    query=query,
                    skill=skill,
                )
            )
        return items

    @staticmethod
    def _build_retrieval_diagnostics(
        coder_result: Dict[str, Any],
        skill_names: List[str],
    ) -> List[Dict[str, Any]]:
        diagnostics: List[Dict[str, Any]] = []
        for skill_name in skill_names:
            info = coder_result.get("per_skill", {}).get(skill_name, {})
            diagnostics.append(
                {
                    "skill": skill_name,
                    "strategy": info.get("strategy", ""),
                    "error": info.get("error", ""),
                    "records": len(info.get("records", []) or []),
                    "output": info.get("output", ""),
                }
            )
        return diagnostics

    @staticmethod
    def _evidence_items_to_retrieved_content(evidence_items) -> List[Dict[str, Any]]:
        records = []
        for item in evidence_items:
            records.append({
                "source": item.source_skill,
                "source_entity": "",
                "source_type": item.source_type,
                "target_entity": "",
                "target_type": "",
                "relationship": item.claim,
                "weight": 1.0,
                "evidence_text": item.snippet,
                "skill_category": item.metadata.get("skill_category", ""),
                "sources": [item.source_locator] if item.source_locator else [],
            })
        return records
