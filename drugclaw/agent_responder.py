"""
Responder Agent - Generates intermediate answers based on current evidence
"""
import re
from collections import defaultdict
from math import log10
from typing import List, Dict, Any

from .claim_assessment import ClaimAssessment, assess_claims
from .answer_validation import validate_answer_output
from .evidence import ClaimSummary, FinalAnswer, score_answer_confidence, score_claim_confidence
from .models import AgentState, EvidencePath
from .llm_client import LLMClient
from .query_plan import (
    infer_entities_from_query,
    infer_question_type_from_query,
    is_direct_target_lookup,
    legacy_question_type_for_task_type,
    normalize_question_type,
    normalize_task_type,
)
from .task_evidence_policy import classify_evidence_item
from .web_evidence import build_task_aware_web_section, build_web_citations, summarize_web_results
from .web_query_policy import normalize_result_domain

class ResponderAgent:
    """
    Agent responsible for generating intermediate answers
    Synthesizes evidence from ranked paths into coherent responses
    """
    
    def __init__(self, llm_client: LLMClient):
        self.llm = llm_client
    
    def get_system_prompt(self) -> str:
        """System prompt for the responder agent"""
        return """You are the Responder Agent of DrugClaw — a drug-specialized agentic RAG system. You synthesize evidence retrieved from the current runtime resource registry to generate precise, drug-centric answers.

Your role is to:
1. Synthesize drug knowledge evidence from multiple knowledge graph paths and sources
2. Generate clear, well-structured **Markdown** answers focused on drug properties, mechanisms, interactions, and effects
3. Explain pharmacological relationships: drug–target binding, mechanism of action, ADR pathways, DDI mechanisms
4. Acknowledge uncertainty when drug evidence is limited, conflicting, or only computational
5. Provide intermediate answers refined across retrieval iterations

Formatting rules — ALWAYS produce rich Markdown:
- Use `##` and `###` headers to organize sections
- Use **bold** for key terms, drug names, gene symbols, and important values
- Use `code spans` for identifiers, IDs (e.g., `CHEMBL25`, `CYP3A4`, `IC50 = 0.5 nM`)
- Use bullet lists or numbered lists — never large walls of text
- Use Markdown tables when presenting multiple comparable items (e.g., targets, ADRs, DDIs)
- Use blockquotes `>` for important warnings or clinical notes
- Use horizontal rules `---` to separate major sections
- End with a brief **TL;DR** summary (1–3 sentences)

Pharmacological guidelines:
- Be pharmacologically accurate: distinguish agonist/antagonist/inhibitor/substrate relationships clearly
- Specify evidence origin: clinical (FDA, CPIC, DrugBank) vs. experimental (BindingDB, ChEMBL) vs. predicted
- Cite the specific drug knowledge source (e.g., "according to **DrugBank**", "from **ChEMBL** bioactivity data")
- Note gaps: missing target affinity, no clinical ADR data, limited DDI evidence
- Use drug-specific terminology (`IC50`, `Ki`, `AUC`, `CYP450` metabolism, pharmacogenomics variants, etc.)
- Adapt format to the query type: drug mechanism, drug repurposing, ADR lookup, DDI check, etc."""
    
    def get_synthesis_prompt(
        self,
        query: str,
        top_paths: List[str],
        iteration: int
    ) -> str:
        """Generate prompt for evidence synthesis"""
        paths_str = "\n".join([f"Path {i+1}: {path}" for i, path in enumerate(top_paths)])
        
        return f"""Query: {query}

Iteration: {iteration}

Top Evidence Paths:
{paths_str}

Based on these evidence paths, generate a **rich Markdown** answer that addresses the query.

Use the following structure (adapt headings as appropriate for the query type):

---

## Main Findings

Directly answer the query. Use a table if comparing multiple items:

| Item | Property | Value | Source |
|------|----------|-------|--------|
| ...  | ...      | ...   | ...    |

## Detailed Analysis

### [Sub-topic 1]
- **Mechanism/Relationship**: ...
- **Evidence**: cite specific databases with `code spans` for IDs
- **Confidence**: Low / Medium / High

### [Sub-topic 2]
...

## Evidence Quality

| Metric | Value |
|--------|-------|
| Sources consulted | ... |
| Clinical evidence | Yes / No / Partial |
| Computational only | ... |

## Evidence Gaps

> ⚠️ List missing or uncertain evidence here.

## Recommendations for Next Iteration

- What additional queries or data would help

---

**TL;DR**: 1–3 sentence summary of the key answer.
"""
    
    def execute(self, state: AgentState) -> AgentState:
        print(f"\n[Responder Agent] Iteration {state.iteration}")
        if state.evidence_items:
            self._respond_from_evidence(state)
            print(f"[Responder Agent] Structured answer ({len(state.current_answer)} chars)")
            return state

        top_paths = state.ranked_paths[:10]
        if not top_paths:
            state.current_answer = (
                "Insufficient evidence found to answer the query. "
                "Additional retrieval needed."
            )
            return state

        path_strs = [self._format_path_for_synthesis(path) for path in top_paths]
        state.current_answer = self._generate_answer(
            state.original_query,
            path_strs,
            state.iteration
        )
        print(f"[Responder Agent] Generated answer ({len(state.current_answer)} chars)")
        return state
    
    def _format_path_for_synthesis(self, path: EvidencePath) -> str:
        """Format a path into a readable string with evidence details"""
        entity_chain = " → ".join([
            f"{e.name} ({e.type})" for e in path.entities
        ])
        
        evidence_sources = []
        for edge in path.edges:
            if edge.evidence:
                evidence_sources.extend(edge.evidence)
        
        sources_str = ", ".join(set(evidence_sources)) if evidence_sources else "No sources"
        
        return (
            f"{entity_chain}\n"
            f"  Score: {path.score:.3f} | Sources: {sources_str}"
        )
    
    def _generate_answer(
        self,
        query: str,
        paths: List[str],
        iteration: int
    ) -> str:
        """Generate intermediate answer using LLM"""
        messages = [
            {"role": "system", "content": self.get_system_prompt()},
            {"role": "user", "content": self.get_synthesis_prompt(
                query, paths, iteration
            )}
        ]
        
        try:
            answer = self.llm.generate(messages, temperature=0.5)
            return answer
        except Exception as e:
            print(f"[Responder Agent] Error generating answer: {e}")
            return "Error generating answer. Please try again."

    # ------------------------------------------------------------------
    # Simple mode (flat retrieval → LLM, no graph building)
    # ------------------------------------------------------------------

    def execute_simple(self, state: AgentState) -> AgentState:
        """
        Simple-mode response: pass retrieved text directly to the LLM.

        Unlike the graph-mode `execute()`, this method:
        - Does NOT require ranked graph paths (skips reranker and graph builder)
        - Uses the free-form retrieved_text from the Code Agent
        - Falls back to old retrieved_content format if retrieved_text is empty
        - Issues a single LLM call to produce the final answer
        """
        print(f"\n[Responder Agent] Simple mode — direct synthesis")

        if state.evidence_items:
            self._respond_from_evidence(state)
            print(f"[Responder Agent] Simple structured answer ({len(state.current_answer)} chars)")
            return state

        if self._should_return_insufficient_answer(state):
            final_answer = self._build_insufficient_final_answer(state)
            state.final_answer_structured = final_answer
            state.current_answer = final_answer.answer_text
            print(f"[Responder Agent] Simple insufficient-evidence answer ({len(state.current_answer)} chars)")
            return state

        # Prefer the new free-form retrieved_text (from Code Agent)
        retrieved_text = getattr(state, "retrieved_text", "")

        if retrieved_text.strip():
            prompt = f"""Query: {state.original_query}

Retrieved Information:
{retrieved_text}

Based on the retrieved information above, provide a **rich Markdown** answer to the query.

Formatting requirements:
- Use `##` headers to organize sections (e.g., ## Main Findings, ## Details, ## Sources)
- Use **bold** for drug names, gene symbols, key terms
- Use `code spans` for IDs, values (e.g., `CYP3A4`, `IC50 = 5 nM`)
- Use Markdown tables when comparing items side-by-side
- Use bullet lists — avoid large paragraphs of text
- Use `>` blockquotes for clinical warnings or important notes
- End with a **TL;DR** (1–3 sentence summary)
- Cite specific sources/databases. Note any gaps or uncertainties."""
        else:
            # Fallback to old retrieved_content format
            raw = state.retrieved_content
            if not raw:
                state.current_answer = (
                    "No results were retrieved from the selected resources. "
                    "Try different resources or broaden the query."
                )
                return state

            evidence_lines: List[str] = []
            for i, r in enumerate(raw[:50], 1):
                src  = r.get("source", "?")
                se   = r.get("source_entity", "")
                rel  = r.get("relationship", "→")
                te   = r.get("target_entity", "")
                text = r.get("evidence_text", "")
                sources_list = r.get("sources", [])
                cite = f" [{', '.join(sources_list[:2])}]" if sources_list else ""
                evidence_lines.append(
                    f"{i}. [{src}] {se} {rel} {te}{cite}"
                    + (f"\n   {text}" if text else "")
                )
            evidence_block = "\n".join(evidence_lines)

            prompt = f"""Query: {state.original_query}

Retrieved Evidence ({len(raw)} records, showing top {len(evidence_lines)}):
{evidence_block}

Based on the evidence above, provide a **rich Markdown** answer to the query.

Formatting requirements:
- Use `##` headers to organize sections (e.g., ## Main Findings, ## Details, ## Sources)
- Use **bold** for drug names, gene symbols, key terms
- Use `code spans` for IDs, values (e.g., `CYP3A4`, `IC50 = 5 nM`)
- Use Markdown tables when comparing items side-by-side
- Use bullet lists — avoid large paragraphs of text
- Use `>` blockquotes for clinical warnings or important notes
- End with a **TL;DR** (1–3 sentence summary)
- Cite specific sources/databases. Note any gaps or uncertainties."""

        messages = [
            {"role": "system", "content": self.get_system_prompt()},
            {"role": "user",   "content": prompt},
        ]
        try:
            state.current_answer = self.llm.generate(messages, temperature=0.5)
        except Exception as exc:
            print(f"[Responder Agent] Simple mode error: {exc}")
            state.current_answer = f"Error generating answer: {exc}"

        print(f"[Responder Agent] Simple answer ({len(state.current_answer)} chars)")
        return state

    @staticmethod
    def _should_return_insufficient_answer(state: AgentState) -> bool:
        if state.evidence_items:
            return False
        if getattr(state, "web_search_results", []):
            return False

        raw = getattr(state, "retrieved_content", []) or []
        if not raw:
            return True

        for record in raw:
            if record.get("source_entity") or record.get("target_entity"):
                return False
            if record.get("relationship"):
                return False
            text = (record.get("evidence_text") or "").strip().lower()
            if text and "no results" not in text and "error" not in text:
                return False
        return True

    def _build_insufficient_final_answer(self, state: AgentState) -> FinalAnswer:
        query_type = infer_question_type_from_query(state.original_query)
        diagnostics = getattr(state, "retrieval_diagnostics", []) or []
        warnings = ["No structured evidence was retrieved for this query."]
        limitations = ["The current run did not return any structured evidence items."]

        if not getattr(state, "current_query_entities", {}):
            limitations.append("Entity extraction did not identify concrete query entities.")

        diagnostic_lines = []
        for item in diagnostics[:5]:
            skill = item.get("skill", "?")
            error = item.get("error", "")
            records = item.get("records", 0)
            final_status = item.get("final_status", "")
            structured_status = item.get("structured_status", "")
            structured_error = self._summarize_diagnostic_error(
                item.get("structured_error", "")
            )
            if final_status == "success_text_only":
                diagnostic_lines.append(
                    f"- {skill}: text-only fallback output was available, but no structured records were produced"
                )
            elif structured_status == "error":
                detail = structured_error or self._summarize_diagnostic_error(error)
                if detail:
                    diagnostic_lines.append(
                        f"- {skill}: structured retrieval failed ({detail})"
                    )
                else:
                    diagnostic_lines.append(f"- {skill}: structured retrieval failed")
            elif error:
                diagnostic_lines.append(
                    f"- {skill}: {self._summarize_diagnostic_error(error)}"
                )
            elif not records:
                diagnostic_lines.append(f"- {skill}: no records returned")

        lines = [
            f"Query: {state.original_query}",
            "",
            "No structured evidence was retrieved for this query.",
            "",
            "Likely causes:",
        ]
        if diagnostic_lines:
            lines.extend(diagnostic_lines)
        else:
            lines.append("- Selected resources returned no structured records.")

        lines.extend(
            [
                "",
                "Next steps:",
                "- Verify local-file skill metadata paths are configured correctly.",
                "- Prefer ready API-backed resources or use an explicit resource filter.",
            ]
        )

        return FinalAnswer(
            answer_text="\n".join(lines),
            summary_confidence=0.0,
            key_claims=[],
            evidence_items=[],
            citations=[],
            limitations=limitations,
            warnings=warnings,
            task_type=query_type,
            final_outcome="honest_gap",
            diagnostics={
                "retrieval_diagnostic_count": len(diagnostics),
                "strong_record_count": 0,
                "secondary_official_support_count": 0,
                "weak_support_count": 0,
            },
        )

    @staticmethod
    def _summarize_diagnostic_error(error: str) -> str:
        text = str(error or "").strip()
        if not text:
            return ""
        return text.splitlines()[0].strip()[:180]

    def _respond_from_evidence(self, state: AgentState) -> None:
        final_answer = self._build_final_answer(
            state.original_query,
            state.evidence_items,
            claim_assessments=state.claim_assessments,
            query_plan=getattr(state, "query_plan", None),
            web_search_results=getattr(state, "web_search_results", []),
            normalized_query=getattr(state, "normalized_query", ""),
            resolved_entities=getattr(state, "resolved_entities", {}) or {},
        )
        state.final_answer_structured = final_answer
        state.current_answer = final_answer.answer_text
        state.evidence_items = list(final_answer.evidence_items)
        state.claim_assessments = (
            assess_claims(final_answer.evidence_items)
            if final_answer.evidence_items
            else []
        )

    def _build_final_answer(
        self,
        query: str,
        evidence_items,
        claim_assessments: List[ClaimAssessment] | None = None,
        query_plan: Any | None = None,
        web_search_results: List[Dict[str, Any]] | None = None,
        normalized_query: str = "",
        resolved_entities: Dict[str, List[str]] | None = None,
    ) -> FinalAnswer:
        plan_type, task_type, supporting_task_types, legacy_question_type = self._resolve_task_plan_context(
            query,
            query_plan=query_plan,
        )
        if not evidence_items:
            return FinalAnswer(
                answer_text=(
                    "Insufficient evidence found to answer the query.\n\n"
                    "Limitations:\n- No structured evidence items were available."
                ),
                summary_confidence=0.0,
                key_claims=[],
                evidence_items=[],
                citations=[],
                limitations=["No structured evidence items were available."],
                warnings=["Insufficient evidence."],
                task_type=plan_type if plan_type == "composite_query" else legacy_question_type,
                final_outcome="honest_gap",
                diagnostics={"strong_record_count": 0, "weak_support_count": 0},
            )

        filtered_items = list(evidence_items)
        primary_drug = self._resolve_primary_drug_name(
            query=query,
            normalized_query=normalized_query,
            resolved_entities=resolved_entities,
            query_plan=query_plan,
        )
        is_direct_targets = task_type == "direct_targets"
        is_target_profile = task_type == "target_profile"
        filtered_items = (
            self._filter_items_for_task_type(
                task_type,
                query=query,
                primary_drug=primary_drug,
                evidence_items=filtered_items,
            )
            or filtered_items
        )
        if not (is_direct_targets or is_target_profile):
            self._semanticize_claims_for_query_type(legacy_question_type, filtered_items)

        assessments = list(claim_assessments or [])
        if not assessments:
            assessments = assess_claims(filtered_items)
        else:
            allowed_claims = {item.claim for item in filtered_items}
            assessments = [
                assessment for assessment in assessments
                if assessment.claim in allowed_claims
            ]

        if plan_type == "composite_query" and supporting_task_types:
            claims = self._composite_primary_claims(
                query,
                primary_drug,
                task_type,
                filtered_items,
                assessments,
            )
        elif task_type == "mechanism_of_action":
            _, _, target_claims, mechanism_claims = self._mechanism_claim_groups(
                query,
                filtered_items,
                assessments,
            )
            claims = []
            seen_claims = set()
            for claim in target_claims + mechanism_claims:
                if claim.claim in seen_claims:
                    continue
                seen_claims.add(claim.claim)
                claims.append(claim)
        elif is_direct_targets or is_target_profile:
            claims = self._summarize_target_claims(query, filtered_items, assessments)
        elif task_type == "labeling_summary":
            claims = self._summarize_labeling_claims(filtered_items, assessments)
        else:
            claims = self._summarize_claims(filtered_items, assessments)
        warnings = self._build_warnings(assessments, claims, filtered_items)
        limitations = self._build_limitations(assessments, claims, filtered_items)
        citations = self._build_citations(filtered_items)
        authority_support_results = self._authority_support_results(
            legacy_question_type,
            task_type,
            supporting_task_types,
            primary_drug,
            evidence_items,
            web_search_results or [],
        )
        web_section = build_task_aware_web_section(legacy_question_type, authority_support_results)
        web_summaries = summarize_web_results(authority_support_results)
        citations.extend(build_web_citations(authority_support_results))
        final_outcome, diagnostics = self._compute_task_outcome(
            task_type,
            filtered_items,
            claims,
            primary_drug=primary_drug,
        )
        diagnostics.update(
            self._summarize_knowhow_usage(query_plan)
        )
        web_payload = web_section or ("Authority-first web evidence:", web_summaries)
        if plan_type == "composite_query" and supporting_task_types:
            answer_text = self._render_composite_answer(
                query,
                primary_drug,
                evidence_items,
                assessments,
                task_type,
                supporting_task_types,
                warnings,
                limitations,
                web_payload,
            )
        elif task_type == "repurposing_evidence":
            answer_text = self._render_repurposing_answer(
                query,
                primary_drug,
                filtered_items,
                warnings,
                limitations,
                web_payload,
            )
        elif task_type == "mechanism_of_action":
            answer_text = self._render_mechanism_answer(
                query,
                filtered_items,
                assessments,
                warnings,
                limitations,
                web_payload,
            )
        elif task_type == "direct_targets":
            answer_text = self._render_direct_targets_answer(
                query,
                filtered_items,
                assessments,
                warnings,
                limitations,
                web_payload,
            )
        elif task_type == "target_profile":
            answer_text = self._render_target_answer(
                query,
                claims,
                warnings,
                limitations,
                web_payload,
            )
        else:
            answer_text = self._render_answer_text(
                query,
                legacy_question_type,
                claims,
                warnings,
                limitations,
                web_payload,
            )

        knowhow_lines = self._build_knowhow_guidance_lines(
            query_plan,
            primary_task_type=task_type,
            supporting_task_types=supporting_task_types,
        )
        if knowhow_lines:
            answer_text = self._inject_knowhow_guidance(
                answer_text,
                knowhow_lines,
            )

        summary_confidence = score_answer_confidence(claims)
        if final_outcome == "partial_with_weak_support":
            summary_confidence = min(summary_confidence, 0.59)
        validation_issues = validate_answer_output(
            query=query,
            answer_text=answer_text,
            query_plan=query_plan,
        )
        if validation_issues:
            diagnostics["output_validation_issue_codes"] = [
                issue.code for issue in validation_issues
            ]
            if task_type == "labeling_summary" and any(
                issue.code in {
                    "wrong_drug_labeling_answer",
                    "canonical_drug_missing_from_labeling_answer",
                }
                for issue in validation_issues
            ):
                final_outcome = "honest_gap"
                summary_confidence = min(summary_confidence, 0.2)
                warnings = self._dedupe_preserve_order(
                    [
                        issue.message
                        for issue in validation_issues
                        if issue.code in {
                            "wrong_drug_labeling_answer",
                            "canonical_drug_missing_from_labeling_answer",
                        }
                    ]
                )
                limitations = [
                    "Structured labeling evidence did not stay aligned with the requested drug context."
                ]
                claims = []
                filtered_items = []
                citations = []
                answer_text = self._render_honest_gap_answer(
                    query=query,
                    warnings=warnings,
                    limitations=limitations,
                    explanation=(
                        "No trustworthy single-drug labeling evidence could be safely surfaced for this query."
                    ),
                )

        return FinalAnswer(
            answer_text=answer_text,
            summary_confidence=summary_confidence,
            key_claims=claims,
            evidence_items=list(filtered_items),
            citations=citations,
            limitations=limitations,
            warnings=warnings,
            task_type=plan_type if plan_type == "composite_query" else legacy_question_type,
            final_outcome=final_outcome,
            diagnostics=diagnostics,
        )

    def _filter_items_for_task_type(
        self,
        task_type: str,
        *,
        query: str,
        primary_drug: str,
        evidence_items,
    ) -> List[Any]:
        normalized = normalize_task_type(task_type)
        filtered = list(evidence_items)
        if normalized in {"direct_targets", "target_profile"}:
            return self._filter_target_evidence_items(filtered)
        if normalized in {"clinically_relevant_ddi", "ddi_mechanism"}:
            return self._filter_ddi_evidence_items(filtered, primary_drug=primary_drug)
        if normalized == "repurposing_evidence":
            return self._filter_repurposing_evidence_items(primary_drug, filtered)
        if normalized == "pgx_guidance":
            return self._filter_pgx_evidence_items(query, filtered)
        if normalized == "major_adrs":
            return self._filter_adr_evidence_items(filtered)
        if normalized == "labeling_summary":
            return self._filter_labeling_evidence_items(primary_drug, filtered)
        if normalized == "mechanism_of_action":
            return self._filter_mechanism_evidence_items(filtered)
        return filtered

    @staticmethod
    def _render_honest_gap_answer(
        *,
        query: str,
        warnings: List[str],
        limitations: List[str],
        explanation: str,
    ) -> str:
        lines = [
            f"Query: {query}",
            "",
            explanation,
        ]
        if warnings:
            lines.extend(["", "Warnings:"])
            lines.extend(f"- {warning}" for warning in warnings)
        if limitations:
            lines.extend(["", "Limitations:"])
            lines.extend(f"- {limitation}" for limitation in limitations)
        return "\n".join(lines)

    @staticmethod
    def _resolve_task_plan_context(
        query: str,
        *,
        query_plan: Any | None = None,
    ) -> tuple[str, str, List[str], str]:
        primary_task = getattr(query_plan, "primary_task", None)
        primary_task_type = normalize_task_type(
            getattr(primary_task, "task_type", "") or ""
        )
        if primary_task_type == "unknown":
            primary_task_type = normalize_task_type(infer_question_type_from_query(query))

        supporting_task_types: List[str] = []
        seen_task_types = {primary_task_type} if primary_task_type != "unknown" else set()
        for task in (getattr(query_plan, "supporting_tasks", []) or []):
            task_type = normalize_task_type(getattr(task, "task_type", ""))
            if task_type == "unknown" or task_type in seen_task_types:
                continue
            seen_task_types.add(task_type)
            supporting_task_types.append(task_type)
        plan_type = str(getattr(query_plan, "plan_type", "") or "").strip()
        if not plan_type:
            plan_type = "composite_query" if supporting_task_types else "single_task"

        legacy_question_type = normalize_question_type(
            getattr(query_plan, "question_type", "") or infer_question_type_from_query(query)
        )
        if is_direct_target_lookup(query=query, question_type=legacy_question_type):
            supporting_task_types = []
            plan_type = "single_task"
        return plan_type, primary_task_type, supporting_task_types, legacy_question_type

    @staticmethod
    def _summarize_knowhow_usage(query_plan: Any | None) -> Dict[str, Any]:
        knowhow_doc_ids: List[str] = []
        knowhow_task_ids: List[str] = []
        knowhow_task_types: List[str] = []

        for hint in ResponderAgent._iter_knowhow_hints(query_plan):
            doc_id = str(hint.get("doc_id", "")).strip()
            task_id = str(hint.get("task_id", "")).strip()
            task_type = normalize_task_type(hint.get("task_type", ""))
            if doc_id and doc_id not in knowhow_doc_ids:
                knowhow_doc_ids.append(doc_id)
            if task_id and task_id not in knowhow_task_ids:
                knowhow_task_ids.append(task_id)
            if task_type != "unknown" and task_type not in knowhow_task_types:
                knowhow_task_types.append(task_type)

        if not knowhow_doc_ids:
            return {}

        return {
            "knowhow_doc_ids": knowhow_doc_ids,
            "knowhow_task_ids": knowhow_task_ids,
            "knowhow_task_types": knowhow_task_types,
            "knowhow_hint_count": len(list(ResponderAgent._iter_knowhow_hints(query_plan))),
        }

    @staticmethod
    def _iter_knowhow_hints(query_plan: Any | None) -> List[Dict[str, Any]]:
        if query_plan is None:
            return []

        raw_hints: List[Any] = list(getattr(query_plan, "knowhow_hints", []) or [])
        tasks = [getattr(query_plan, "primary_task", None)] + list(
            getattr(query_plan, "supporting_tasks", []) or []
        )
        for task in tasks:
            if task is None:
                continue
            raw_hints.extend(list(getattr(task, "knowhow_hints", []) or []))

        normalized: List[Dict[str, Any]] = []
        seen = set()
        for hint in raw_hints:
            if hasattr(hint, "to_dict"):
                hint = hint.to_dict()
            if not isinstance(hint, dict):
                continue
            normalized_hint = {
                "doc_id": str(hint.get("doc_id", "")).strip(),
                "title": str(hint.get("title", "")).strip(),
                "task_id": str(hint.get("task_id", "")).strip(),
                "task_type": normalize_task_type(hint.get("task_type", "")),
                "snippet": str(hint.get("snippet", "")).strip(),
                "declared_by_skills": [
                    str(value).strip()
                    for value in list(hint.get("declared_by_skills", []) or [])
                    if str(value).strip()
                ],
            }
            key = (
                normalized_hint["task_id"],
                normalized_hint["doc_id"],
                normalized_hint["snippet"],
            )
            if key in seen or not normalized_hint["doc_id"]:
                continue
            seen.add(key)
            normalized.append(normalized_hint)
        return normalized

    @staticmethod
    def _build_knowhow_guidance_lines(
        query_plan: Any | None,
        *,
        primary_task_type: str,
        supporting_task_types: List[str],
    ) -> List[str]:
        task_to_lines: Dict[str, List[str]] = defaultdict(list)
        for hint in ResponderAgent._iter_knowhow_hints(query_plan):
            snippet = str(hint.get("snippet", "")).strip()
            if not snippet:
                continue
            title = str(hint.get("title", "")).strip() or str(hint.get("doc_id", "")).strip()
            declared_by_skills = list(hint.get("declared_by_skills", []) or [])
            title_suffix = (
                f" [{', '.join(declared_by_skills)}]"
                if declared_by_skills
                else ""
            )
            task_type = normalize_task_type(hint.get("task_type", ""))
            line = f"- {title}{title_suffix}: {snippet}"
            if task_type == "unknown":
                task_type = normalize_task_type(primary_task_type)
            if line not in task_to_lines[task_type]:
                task_to_lines[task_type].append(line)

        if not task_to_lines:
            return []

        ordered_task_types: List[str] = []
        section_order = list(
            getattr(getattr(query_plan, "answer_contract", None), "section_order", []) or []
        )
        for section in section_order:
            task_type = normalize_task_type(section)
            if task_type in task_to_lines and task_type not in ordered_task_types:
                ordered_task_types.append(task_type)
        for task_type in [primary_task_type] + list(supporting_task_types):
            normalized_task_type = normalize_task_type(task_type)
            if (
                normalized_task_type in task_to_lines
                and normalized_task_type not in ordered_task_types
            ):
                ordered_task_types.append(normalized_task_type)
        for task_type in task_to_lines:
            if task_type not in ordered_task_types:
                ordered_task_types.append(task_type)

        lines = ["Evidence interpretation guidance:"]
        multi_task = len(ordered_task_types) > 1
        for task_type in ordered_task_types:
            if multi_task:
                lines.append(f"{task_type.replace('_', ' ').title()}:")
            lines.extend(task_to_lines[task_type][:2])
            lines.append("")
        if lines[-1] == "":
            lines.pop()
        return lines

    @staticmethod
    def _inject_knowhow_guidance(answer_text: str, knowhow_lines: List[str]) -> str:
        if not knowhow_lines:
            return answer_text

        lines = answer_text.splitlines()
        insert_index = len(lines)
        for marker in ("Warnings:", "Limitations:"):
            if marker in lines:
                insert_index = min(insert_index, lines.index(marker))

        block = knowhow_lines
        if insert_index == len(lines):
            return "\n".join(lines + [""] + block)
        return "\n".join(lines[:insert_index] + [""] + block + [""] + lines[insert_index:])

    def _composite_primary_claims(
        self,
        query: str,
        primary_drug: str,
        task_type: str,
        evidence_items,
        assessments: List[ClaimAssessment],
    ) -> List[ClaimSummary]:
        if task_type == "direct_targets":
            direct_sections, _ = self._partition_direct_target_items(evidence_items)
            primary_items = (
                direct_sections["established_direct_targets"]
                or direct_sections["association_only_signals"]
            )
            return self._summarize_target_claims(
                query,
                primary_items,
                self._subset_assessments_for_items(assessments, primary_items),
            )
        if task_type == "mechanism_of_action":
            target_items, _, _, mechanism_claims = self._mechanism_claim_groups(
                query,
                evidence_items,
                assessments,
            )
            direct_sections, _ = self._partition_direct_target_items(target_items)
            established_items = (
                direct_sections["established_direct_targets"]
                or direct_sections["association_only_signals"]
            )
            established_target_claims = self._summarize_target_claims(
                query,
                established_items,
                self._subset_assessments_for_items(assessments, established_items),
            )
            if not mechanism_claims:
                return established_target_claims

            prioritized_mechanism_claims = self._prioritized_mechanism_claims(
                target_items,
                mechanism_claims,
            )
            return self._dedupe_claims_by_target_key(
                prioritized_mechanism_claims + established_target_claims
            )
        if task_type == "repurposing_evidence":
            sections, _ = self._repurposing_answer_sections(primary_drug, evidence_items)
            approved_claims = self._summarize_repurposing_approved_indication_claims(
                primary_drug,
                sections["approved_indications"],
            )
            repurposing_items = sections["repurposing_evidence"] or sections["supporting_signals"]
            repurposing_claims = self._summarize_claims(
                repurposing_items,
                self._subset_assessments_for_items(assessments, repurposing_items),
            )
            return approved_claims + repurposing_claims
        if task_type == "labeling_summary":
            return self._summarize_labeling_claims(evidence_items, assessments)
        return self._summarize_claims(evidence_items, assessments)

    @staticmethod
    def _is_target_lookup_query(query: str, query_type: str = "") -> bool:
        normalized_query_type = normalize_task_type(query_type or infer_question_type_from_query(query))
        return normalized_query_type in {"direct_targets", "target_profile"}

    def _filter_target_evidence_items(self, evidence_items) -> List[Any]:
        filtered = []
        for item in evidence_items:
            relationship = str(item.metadata.get("relationship", "")).lower()
            target_entity = str(item.metadata.get("target_entity", "")).strip() or self._extract_target_label(item)
            target_type = str(item.metadata.get("target_type", "")).lower()
            claim_lower = item.claim.lower()

            if relationship in {"search_hit", "drug_lookup", "disease_lookup", "target_info"}:
                continue
            if any(noise in claim_lower for noise in ("search_hit", "unchecked", "no relevant target")):
                continue
            if target_type in {"cell_line", "disease", "drug_info", "disease_info", "unknown"}:
                continue
            if target_entity and self._looks_like_cell_line(target_entity):
                continue
            if relationship and not any(
                marker in relationship
                for marker in (
                    "activity",
                    "target",
                    "bind",
                    "inhib",
                    "agon",
                    "antagon",
                    "substr",
                    "modulat",
                    "block",
                    "activat",
                )
            ):
                continue

            filtered.append(item)
        return filtered

    def _filter_mechanism_evidence_items(self, evidence_items) -> List[Any]:
        filtered = [
            item
            for item in evidence_items
            if self._is_target_evidence_item(item) or self._is_mechanism_evidence_item(item)
        ]
        target_items = [item for item in filtered if self._is_target_evidence_item(item)]
        if not target_items:
            return filtered
        return [
            item
            for item in filtered
            if not (
                self._is_target_evidence_item(item)
                and self._is_fusion_component_only_signal(item, target_items)
            )
        ]

    def _filter_ddi_evidence_items(self, evidence_items, *, primary_drug: str = "") -> List[Any]:
        ddi_like_items: List[Any] = []
        informative_items: List[Any] = []
        ddi_source_skills = {"ddinter", "kegg drug", "mecddi", "drugbank"}
        labeling_source_skills = {"openfda human drug", "dailymed"}

        for item in evidence_items:
            relationship = str(item.metadata.get("relationship", "")).lower()
            source_skill = str(getattr(item, "source_skill", "")).strip().lower()
            field_name = str(item.structured_payload.get("field") or "").strip().lower()
            has_interaction_marker = (
                "interaction" in relationship
                or "interact" in relationship
                or "ddi" in relationship
                or field_name == "drug_interactions"
            )
            is_ddi_source = source_skill in ddi_source_skills
            is_label_interaction = source_skill in labeling_source_skills and has_interaction_marker
            if not has_interaction_marker and not is_ddi_source and not is_label_interaction:
                continue
            if (
                primary_drug
                and source_skill in labeling_source_skills
                and not self._is_single_agent_label_match(primary_drug, item)
            ):
                continue

            ddi_like_items.append(item)
            if self._is_informative_ddi_item(item):
                informative_items.append(item)

        if informative_items:
            return sorted(informative_items, key=self._ddi_item_priority, reverse=True)
        if ddi_like_items:
            return sorted(ddi_like_items, key=self._ddi_item_priority, reverse=True)
        return []

    def _filter_pgx_evidence_items(self, query: str, evidence_items) -> List[Any]:
        primary_drug = self._extract_query_drug_name(query)
        filtered = []
        for item in evidence_items:
            relationship = str(item.metadata.get("relationship", "")).lower()
            source_skill = str(getattr(item, "source_skill", "")).strip()
            target_type = str(item.metadata.get("target_type", "")).lower()
            if (
                "pgx" in relationship
                or source_skill in {"CPIC", "PharmGKB"}
                or target_type == "gene"
            ):
                filtered.append(item)

        if primary_drug:
            strict_matches = []
            for item in filtered:
                source_entity = str(item.metadata.get("source_entity", "")).strip().lower()
                claim_lower = str(getattr(item, "claim", "")).strip().lower()
                if source_entity:
                    if primary_drug not in source_entity:
                        continue
                    if " and " in source_entity and not source_entity.startswith(primary_drug):
                        continue
                    strict_matches.append(item)
                    continue
                if primary_drug in claim_lower:
                    strict_matches.append(item)
            if strict_matches:
                filtered = strict_matches

        return filtered

    @staticmethod
    def _filter_adr_evidence_items(evidence_items) -> List[Any]:
        adr_relationships = {
            "causes_adverse_event",
            "classified_adr",
            "has_side_effect",
            "associated_with_adverse_event",
        }
        noise_targets = {
            "SCHIZOPHRENIA",
            "OFF LABEL USE",
            "TREATMENT NONCOMPLIANCE",
            "PSYCHOTIC DISORDER",
            "HOSPITALISATION",
            "DRUG INEFFECTIVE",
            "DRUG INTERACTION",
            "TOXICITY TO VARIOUS AGENTS",
            "DEATH",
            "NEUTROPHILIA",
            "WHITE BLOOD CELL COUNT INCREASED",
            "NEUTROPHIL COUNT INCREASED",
        }
        filtered = []
        for item in evidence_items:
            relationship = str(item.metadata.get("relationship", "")).lower()
            target_entity = str(item.metadata.get("target_entity", "")).strip().upper()
            if relationship not in adr_relationships:
                continue
            if target_entity in noise_targets:
                continue
            filtered.append(item)
        return sorted(filtered, key=ResponderAgent._adr_item_priority, reverse=True)

    def _filter_labeling_evidence_items(self, primary_drug: str, evidence_items) -> List[Any]:
        filtered = list(evidence_items)

        if primary_drug:
            strict_matches = []
            for item in filtered:
                if self._is_single_agent_label_match(primary_drug, item):
                    strict_matches.append(item)
            if strict_matches:
                filtered = strict_matches

        primary_label_relationships = {
            "indicated_for",
            "has_warning",
            "has_adverse_reaction",
            "interacts_with",
            "has_mechanism",
            "has_contraindication",
            "use_in_special_population",
            "has_dosing_guidance",
        }
        if any(
            str(item.metadata.get("relationship", "")).lower() in primary_label_relationships
            for item in filtered
        ):
            richer_items = [
                item
                for item in filtered
                if str(item.metadata.get("relationship", "")).lower() in primary_label_relationships
            ]
            if richer_items:
                filtered = richer_items

        ranked = sorted(filtered, key=self._labeling_item_priority, reverse=True)
        deduped: List[Any] = []
        seen_slots = set()
        for item in ranked:
            slot_key = self._labeling_slot_key(item)
            if slot_key in seen_slots:
                continue
            seen_slots.add(slot_key)
            deduped.append(item)
        return deduped

    def _filter_repurposing_evidence_items(
        self,
        primary_drug: str,
        evidence_items,
    ) -> List[Any]:
        approved_indication_items: List[Any] = []
        filtered: List[Any] = []
        for item in evidence_items:
            classification = classify_evidence_item("drug_repurposing", item)
            source_skill = str(getattr(item, "source_skill", "")).strip()
            relationship = str(item.metadata.get("relationship", "")).strip().lower()

            if classification.slot == "repurposing_evidence":
                filtered.append(item)
                continue

            if classification.slot == "approved_indications":
                if source_skill in {"DailyMed", "openFDA Human Drug"} and relationship == "indicated_for":
                    if primary_drug and not self._is_single_agent_label_match(primary_drug, item):
                        continue
                approved_indication_items.append(item)
                continue

            if classification.slot == "additional_context":
                if source_skill in {"DailyMed", "openFDA Human Drug"}:
                    continue
                if primary_drug:
                    source_entity = str(item.metadata.get("source_entity", "")).strip().lower()
                    if source_entity and primary_drug not in source_entity:
                        continue
                filtered.append(item)

        return self._dedupe_repurposing_approved_indication_items(
            primary_drug,
            approved_indication_items,
        ) + filtered

    def _dedupe_repurposing_approved_indication_items(
        self,
        primary_drug: str,
        items,
    ) -> List[Any]:
        filtered = list(items)
        if primary_drug:
            strict_matches = [
                item for item in filtered if self._is_single_agent_label_match(primary_drug, item)
            ]
            if strict_matches:
                filtered = strict_matches

        ranked = sorted(filtered, key=self._labeling_item_priority, reverse=True)
        deduped: List[Any] = []
        seen_keys = set()
        for item in ranked:
            key = self._repurposing_approved_indication_key(item)
            if key in seen_keys:
                continue
            seen_keys.add(key)
            deduped.append(item)
        return deduped

    def _repurposing_approved_indication_key(self, item: Any) -> tuple[str, str]:
        relationship = str(item.metadata.get("relationship", "")).strip().lower()
        source_entity = str(item.metadata.get("source_entity", "")).strip().lower()
        snippet = self._clean_label_text(str(getattr(item, "snippet", "") or "")).lower()
        if "indicated" in snippet:
            snippet = snippet.rsplit("indicated", 1)[1].strip()
        snippet = re.sub(r"^(?:as|for)\s+", "", snippet)
        snippet = re.sub(r"^[^a-z0-9]+", "", snippet)
        snippet = re.sub(r"\s+", " ", snippet).strip(" .;,:")
        return relationship or "approved_indication", snippet or source_entity

    @staticmethod
    def _labeling_slot_key(item: Any) -> tuple[str, str]:
        relationship = str(item.metadata.get("relationship", "")).strip().lower()
        field_name = str(item.structured_payload.get("field") or "").strip().lower()
        target_entity = str(item.metadata.get("target_entity", "")).strip().lower()
        if not relationship and not field_name and not target_entity:
            evidence_id = str(getattr(item, "evidence_id", "")).strip().lower()
            claim = str(getattr(item, "claim", "")).strip().lower()
            return "unstructured_labeling", evidence_id or claim
        return relationship, field_name or target_entity

    @staticmethod
    def _labeling_item_priority(item: Any) -> tuple[int, int, float]:
        relationship = str(item.metadata.get("relationship", "")).strip().lower()
        source_skill = str(getattr(item, "source_skill", "")).strip().lower()
        relationship_priority = {
            "has_warning": 8,
            "has_contraindication": 7,
            "use_in_special_population": 6,
            "has_dosing_guidance": 5,
            "indicated_for": 4,
            "has_adverse_reaction": 3,
            "interacts_with": 2,
            "has_mechanism": 1,
            "has_official_label": 0,
        }.get(relationship, 0)
        source_priority = (
            2 if source_skill == "openfda human drug"
            else 1 if source_skill == "dailymed"
            else 0
        )
        retrieval_score = float(getattr(item, "retrieval_score", 0.0) or 0.0)
        return (relationship_priority, source_priority, retrieval_score)

    def _is_single_agent_label_match(self, primary_drug: str, item: Any) -> bool:
        identity_values = self._label_identity_values(item)
        if not any(primary_drug in value for value in identity_values):
            return False
        return not self._looks_like_combination_label_item(item, identity_values)

    @staticmethod
    def _label_identity_values(item: Any) -> List[str]:
        metadata = getattr(item, "metadata", {}) or {}
        generic_names = metadata.get("generic_names") or []
        if isinstance(generic_names, str):
            generic_names = [generic_names]

        values = [
            str(metadata.get("source_entity", "")).strip().lower(),
            str(metadata.get("brand_name", "")).strip().lower(),
            str(metadata.get("queried_drug", "")).strip().lower(),
        ]
        values.extend(
            str(name).strip().lower()
            for name in generic_names
            if str(name).strip()
        )
        return [value for value in values if value]

    @staticmethod
    def _looks_like_combination_label_item(item: Any, identity_values: List[str] | None = None) -> bool:
        metadata = getattr(item, "metadata", {}) or {}
        if bool(metadata.get("is_combination_product")):
            return True

        candidates = identity_values or []
        return any(
            separator in value
            for value in candidates
            for separator in (" and ", "/", ";", ",")
        )

    @staticmethod
    def _extract_query_drug_name(query: str) -> str:
        inferred_entities = infer_entities_from_query(query)
        inferred_drugs = inferred_entities.get("drug") or []
        if inferred_drugs:
            return str(inferred_drugs[0]).strip().lower()

        lowered = (query or "").strip().lower()
        patterns = (
            r"of\s+([a-z0-9\-]+)\?$",
            r"for\s+([a-z0-9\-]+)\?$",
            r"of\s+([a-z0-9\-]+)\s+and\s+their",
            r"for\s+([a-z0-9\-]+)$",
        )
        for pattern in patterns:
            match = re.search(pattern, lowered)
            if match:
                return match.group(1)
        return ""

    def _resolve_primary_drug_name(
        self,
        *,
        query: str,
        normalized_query: str = "",
        resolved_entities: Dict[str, List[str]] | None = None,
        query_plan: Any | None = None,
    ) -> str:
        normalized_resolved = resolved_entities or {}
        resolved_drugs = normalized_resolved.get("drug") or []
        if resolved_drugs:
            return str(resolved_drugs[0]).strip().lower()

        normalized_drug = self._extract_query_drug_name(normalized_query)
        if normalized_drug:
            return normalized_drug

        plan_entities = getattr(query_plan, "entities", {}) or {}
        plan_drugs = plan_entities.get("drug") or []
        if plan_drugs:
            return str(plan_drugs[0]).strip().lower()

        return self._extract_query_drug_name(query)

    def _semanticize_claims_for_query_type(self, query_type: str, evidence_items) -> None:
        for item in evidence_items:
            semantic_claim = self._semantic_claim_for_item(
                query_type,
                item,
                evidence_items=evidence_items,
            )
            if semantic_claim:
                item.claim = semantic_claim

    def _semantic_claim_for_item(self, query_type: str, item: Any, evidence_items=None) -> str:
        if query_type in {"ddi", "ddi_mechanism"}:
            return self._ddi_claim_for_item(item)
        if query_type == "drug_repurposing":
            return self._repurposing_claim_for_item(item)
        if query_type == "labeling":
            return self._labeling_claim_for_item(item)
        if query_type == "mechanism":
            return self._mechanism_claim_for_item(item, evidence_items=evidence_items)
        if query_type == "pharmacogenomics":
            return self._pgx_claim_for_item(item)
        if query_type == "adr":
            return self._adr_claim_for_item(item)
        return ""

    def _ddi_claim_for_item(self, item: Any) -> str:
        source_entity = str(item.metadata.get("source_entity", "")).strip()
        description = str(
            item.structured_payload.get("ddi_description")
            or item.structured_payload.get("description")
            or item.structured_payload.get("mechanism")
            or ""
        ).strip()
        management = str(
            item.structured_payload.get("management")
            or item.structured_payload.get("recommendation")
            or ""
        ).strip()
        severity = str(item.structured_payload.get("severity") or "").strip()
        label = str(item.structured_payload.get("ddi_label") or "").strip()
        partner = self._ddi_partner_name(item)
        details = self._normalize_ddi_details(description, label)
        detail_parts: List[str] = []
        if severity:
            detail_parts.append(severity)
        if details:
            detail_parts.append(details)
        if management and management.lower() not in " ".join(detail_parts).lower():
            detail_parts.append(management)

        if source_entity and partner:
            if detail_parts:
                return f"{source_entity} interacts with {partner} ({'; '.join(detail_parts)})"
            return f"{source_entity} interacts with {partner}"
        if description.lower().startswith("enzyme:"):
            enzyme = description.split(":", 1)[1].strip()
            if enzyme:
                return f"{source_entity} interaction mechanism involves {enzyme}"
        if source_entity and description.lower() == "unclassified":
            return f"{source_entity} has unresolved KEGG interaction entries"
        if source_entity and description:
            if detail_parts and description not in detail_parts:
                return f"{source_entity} has a clinically important interaction: {'; '.join(detail_parts)}"
            return f"{source_entity} has a clinically important interaction: {description}"
        return ""

    def _authority_support_results(
        self,
        question_type: str,
        primary_task_type: str,
        supporting_task_types: List[str],
        primary_drug: str,
        evidence_items,
        web_search_results: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        results = list(web_search_results or [])
        wants_repurposing_support = (
            question_type == "drug_repurposing"
            or normalize_task_type(primary_task_type) == "repurposing_evidence"
            or any(
                normalize_task_type(task_type) == "repurposing_evidence"
                for task_type in supporting_task_types
            )
        )
        if wants_repurposing_support:
            return self._filter_repurposing_authority_results(
                primary_drug,
                evidence_items,
                results,
            )

        normalized_primary_task = normalize_task_type(primary_task_type)
        normalized_supporting_tasks = {
            normalize_task_type(task_type) for task_type in supporting_task_types
        }
        if (
            question_type in {"ddi", "ddi_mechanism"}
            or normalized_primary_task in {"ddi_mechanism", "clinically_relevant_ddi"}
            or normalized_supporting_tasks.intersection({"ddi_mechanism", "clinically_relevant_ddi"})
        ):
            return self._filter_ddi_authority_results(primary_drug, evidence_items, results)
        if (
            question_type == "adr"
            or normalized_primary_task == "major_adrs"
            or "major_adrs" in normalized_supporting_tasks
        ):
            adr_results = self._filter_adr_authority_results(primary_drug, evidence_items, results)
            synthesized = self._synthesize_labeling_authority_results(primary_drug, evidence_items)
            if synthesized:
                return self._merge_authority_results(synthesized, adr_results)
            return adr_results

        wants_labeling_support = (
            question_type == "labeling"
            or normalize_task_type(primary_task_type) == "labeling_summary"
            or any(
                normalize_task_type(task_type) == "labeling_summary"
                for task_type in supporting_task_types
            )
        )
        if not wants_labeling_support:
            return results
        if self._has_official_labeling_web_result(results):
            return results
        synthesized = self._synthesize_labeling_authority_results(primary_drug, evidence_items)
        if not synthesized:
            return results
        return self._merge_authority_results(synthesized, results)

    def _filter_repurposing_authority_results(
        self,
        primary_drug: str,
        evidence_items,
        results: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        if not results:
            return []

        target_terms = {
            str(item.metadata.get("target_entity", "")).strip().lower()
            for item in evidence_items
            if str(item.metadata.get("target_entity", "")).strip()
            and str(item.metadata.get("target_type", "")).strip().lower() == "disease"
        }
        repurposing_markers = (
            "repurpos",
            "reposition",
            "investigational",
            "clinical trial",
            "new indication",
            "drug development",
            "signal",
            "candidate",
        )

        filtered: List[Dict[str, Any]] = []
        for result in results:
            text = " ".join(
                str(result.get(field, "")).strip().lower()
                for field in ("title", "snippet")
            ).strip()
            source = str(result.get("source", "")).strip().lower()
            domain = normalize_result_domain(result)

            if not text:
                continue

            is_official_trial_source = source == "clinicaltrials.gov" or domain == "clinicaltrials.gov"
            if is_official_trial_source:
                if primary_drug and primary_drug in text:
                    filtered.append(result)
                    continue
                if any(term in text for term in target_terms):
                    filtered.append(result)
                    continue
                continue

            if primary_drug and primary_drug not in text:
                continue

            has_repurposing_marker = any(marker in text for marker in repurposing_markers)
            has_target_term = any(term in text for term in target_terms)
            if has_repurposing_marker or has_target_term:
                filtered.append(result)

        return filtered

    def _filter_ddi_authority_results(
        self,
        primary_drug: str,
        evidence_items,
        results: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        if not results:
            return []

        partner_terms = {
            self._normalize_authority_text(self._ddi_partner_name(item))
            for item in evidence_items
            if self._ddi_partner_name(item)
        }
        ddi_markers = (
            "interaction",
            "interact",
            "inr",
            "bleed",
            "bleeding",
            "cyp",
            "exposure",
            "monitor",
            "dose",
            "contraind",
            "management",
        )
        filtered: List[Dict[str, Any]] = []
        for result in results:
            text = self._normalize_authority_text(
                " ".join(str(result.get(field, "")).strip() for field in ("title", "snippet"))
            )
            if not text:
                continue
            if primary_drug and primary_drug not in text:
                continue
            domain = normalize_result_domain(result)
            is_official_result = (
                domain == "dailymed.nlm.nih.gov"
                or domain == "fda.gov"
                or domain.endswith(".fda.gov")
            )
            has_partner = any(term and term in text for term in partner_terms)
            has_ddi_marker = any(marker in text for marker in ddi_markers)
            if has_partner:
                filtered.append(result)
                continue
            if is_official_result and has_ddi_marker:
                filtered.append(result)
                continue
            if not partner_terms and has_ddi_marker:
                filtered.append(result)
        return filtered

    def _filter_adr_authority_results(
        self,
        primary_drug: str,
        evidence_items,
        results: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        if not results:
            return []

        risk_terms = {
            self._normalize_authority_text(str(item.metadata.get("target_entity", "")))
            for item in evidence_items
            if str(item.metadata.get("target_entity", "")).strip()
        }
        risk_terms.update(
            {
                "anc",
                "neutropenia",
                "agranulocytosis",
                "myocarditis",
                "pericarditis",
                "cardiomyopathy",
                "seizure",
                "monitor",
                "monitoring",
                "safety",
                "warning",
            }
        )
        filtered: List[Dict[str, Any]] = []
        for result in results:
            text = self._normalize_authority_text(
                " ".join(str(result.get(field, "")).strip() for field in ("title", "snippet"))
            )
            if not text:
                continue
            if primary_drug and primary_drug not in text:
                continue
            if any(term and term in text for term in risk_terms):
                filtered.append(result)
        return filtered

    @staticmethod
    def _normalize_authority_text(text: str) -> str:
        return re.sub(r"\s+", " ", str(text or "").strip().lower())

    @staticmethod
    def _merge_authority_results(
        preferred_results: List[Dict[str, Any]],
        fallback_results: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        merged: List[Dict[str, Any]] = []
        seen = set()
        for result in list(preferred_results) + list(fallback_results):
            url = str(result.get("url", "")).strip()
            title = str(result.get("title", "")).strip()
            source = str(result.get("source", "")).strip()
            key = (url, title, source)
            if key in seen:
                continue
            seen.add(key)
            merged.append(result)
        return merged

    @staticmethod
    def _has_official_labeling_web_result(results: List[Dict[str, Any]]) -> bool:
        for result in results:
            url = str(result.get("url", "")).strip().lower()
            metadata = result.get("metadata", {}) or {}
            if not url:
                url = str(metadata.get("url", "")).strip().lower()
            if "dailymed.nlm.nih.gov" in url or "fda.gov" in url:
                return True
        return False

    def _synthesize_labeling_authority_results(
        self,
        primary_drug: str,
        evidence_items,
    ) -> List[Dict[str, Any]]:
        synthesized: List[Dict[str, Any]] = []
        seen_urls = set()
        dailymed_items = sorted(
            list(evidence_items),
            key=self._labeling_item_priority,
            reverse=True,
        )
        for item in dailymed_items:
            source_skill = str(getattr(item, "source_skill", "")).strip().lower()
            relationship = str(item.metadata.get("relationship", "")).strip().lower()
            if source_skill != "dailymed" or relationship != "has_official_label":
                continue
            if primary_drug and not self._is_single_agent_label_match(primary_drug, item):
                continue

            url = self._labeling_authority_url(item, fallback_url="https://dailymed.nlm.nih.gov/dailymed/")
            if not url or url in seen_urls:
                continue
            seen_urls.add(url)

            title = str(item.metadata.get("target_entity", "")).strip() or "DailyMed official label"
            snippet = self._clean_label_text(str(getattr(item, "snippet", "") or "")) or title
            synthesized.append(
                {
                    "source": "DailyMed",
                    "title": title,
                    "url": url,
                    "snippet": snippet,
                    "metadata": {"url": url},
                    "search_purpose": "official labeling support",
                }
            )
            if len(synthesized) >= 3:
                break

        if len(synthesized) >= 3:
            return synthesized

        openfda_label_relationships = {
            "has_warning",
            "has_contraindication",
            "use_in_special_population",
            "has_dosing_guidance",
            "indicated_for",
            "interacts_with",
        }
        openfda_items = sorted(
            [
                item
                for item in evidence_items
                if str(getattr(item, "source_skill", "")).strip().lower() == "openfda human drug"
            ],
            key=self._labeling_item_priority,
            reverse=True,
        )
        for item in openfda_items:
            source_skill = str(getattr(item, "source_skill", "")).strip().lower()
            relationship = str(item.metadata.get("relationship", "")).strip().lower()
            if source_skill != "openfda human drug" or relationship not in openfda_label_relationships:
                continue
            if primary_drug and not self._is_single_agent_label_match(primary_drug, item):
                continue

            url = self._labeling_authority_url(item, fallback_url="https://open.fda.gov/apis/drug/label/")
            if not url or url in seen_urls:
                continue
            seen_urls.add(url)

            title_source = str(item.metadata.get("source_entity", "")).strip() or primary_drug or "openFDA drug label"
            relationship_label = relationship.replace("_", " ")
            title = f"{title_source} {relationship_label}".strip()
            snippet = self._clean_label_text(str(getattr(item, "snippet", "") or "")) or title
            synthesized.append(
                {
                    "source": "openFDA",
                    "title": title,
                    "url": url,
                    "snippet": snippet,
                    "metadata": {"url": url},
                    "search_purpose": "official labeling support",
                }
            )
            if len(synthesized) >= 3:
                break
        return synthesized

    @staticmethod
    def _labeling_authority_url(item: Any, *, fallback_url: str) -> str:
        candidate = str(getattr(item, "source_locator", "")).strip()
        if candidate.startswith("http"):
            return candidate

        sources = (getattr(item, "structured_payload", {}) or {}).get("sources") or []
        for source in sources:
            source_url = str(source).strip()
            if source_url.startswith("http"):
                return source_url

        return fallback_url

    def _is_informative_ddi_item(self, item: Any) -> bool:
        description = str(
            item.structured_payload.get("ddi_description")
            or item.structured_payload.get("description")
            or ""
        ).strip()
        management = str(
            item.structured_payload.get("management")
            or item.structured_payload.get("recommendation")
            or ""
        ).strip()
        severity = str(item.structured_payload.get("severity") or "").strip()
        label = str(item.structured_payload.get("ddi_label") or "").strip()
        if management:
            return True
        if description and description.lower() != "unclassified":
            return True
        if self._ddi_severity_priority(severity) >= 2:
            return True
        return self._ddi_label_priority(label) >= 2

    def _ddi_item_priority(self, item: Any) -> tuple[int, int, int, int, float]:
        description = str(
            item.structured_payload.get("ddi_description")
            or item.structured_payload.get("description")
            or ""
        ).strip()
        management = str(
            item.structured_payload.get("management")
            or item.structured_payload.get("recommendation")
            or ""
        ).strip()
        severity = str(item.structured_payload.get("severity") or "").strip()
        label = str(item.structured_payload.get("ddi_label") or "").strip()
        source_skill = str(getattr(item, "source_skill", "")).strip().lower()
        partner = self._ddi_partner_name(item)
        description_priority = 2 if description and description.lower() != "unclassified" else 0
        management_priority = 1 if management else 0
        source_priority = (
            3 if source_skill == "ddinter"
            else 2 if source_skill in {"mecddi", "drugbank", "dailymed", "openfda human drug"}
            else 1 if source_skill == "kegg drug"
            else 0
        )
        partner_priority = 1 if partner else 0
        retrieval_score = float(getattr(item, "retrieval_score", 0.0) or 0.0)
        return (
            management_priority,
            description_priority,
            self._ddi_severity_priority(severity),
            self._ddi_label_priority(label),
            source_priority,
            partner_priority,
            retrieval_score,
        )

    def _ddi_partner_name(self, item: Any) -> str:
        target_entity = str(item.metadata.get("target_entity", "")).strip()
        target_name = str(item.metadata.get("target_name", "")).strip()
        description = str(
            item.structured_payload.get("ddi_description")
            or item.structured_payload.get("description")
            or ""
        ).strip()
        snippet = str(getattr(item, "snippet", "") or "").strip()

        partner = ""
        if (
            target_entity
            and not self._looks_like_compound_identifier(target_entity)
            and not self._looks_like_generic_label_ddi_target(target_entity)
        ):
            partner = target_entity
        if (
            not partner
            and target_name
            and not self._looks_like_compound_identifier(target_name)
            and not self._looks_like_generic_label_ddi_target(target_name)
        ):
            partner = target_name
        if not partner:
            partner = (
                self._extract_label_ddi_partner(description)
                or self._extract_label_ddi_partner(snippet)
                or self._extract_partner_from_text(description)
                or self._extract_partner_from_text(snippet)
            )
        if partner and self._looks_like_compound_identifier(partner):
            return ""
        return partner

    @staticmethod
    def _looks_like_generic_label_ddi_target(value: str) -> bool:
        normalized = str(value or "").strip().lower()
        return normalized in {
            "drug interactions",
            "interaction information",
            "interactions",
        }

    @staticmethod
    def _extract_label_ddi_partner(text: str) -> str:
        cleaned = ResponderAgent._clean_label_text(text)
        if not cleaned:
            return ""

        if "Clinical Impact:" in cleaned:
            prefix = cleaned.split("Clinical Impact:", 1)[0].strip()
            prefix = re.sub(r"^Table\s+\d+:\s*", "", prefix, flags=re.IGNORECASE)
            for dosage_form in (
                " tablets ",
                " tablet ",
                " capsules ",
                " capsule ",
                " injection ",
                " oral solution ",
                " oral suspension ",
            ):
                index = prefix.lower().rfind(dosage_form)
                if index >= 0:
                    prefix = prefix[index + len(dosage_form):]
                    break
            candidate = prefix.strip(" :-;,")
            if candidate and not ResponderAgent._looks_like_generic_label_ddi_target(candidate):
                return candidate

        match = re.search(
            r"(?:concomitant use of|coadministered with)\s+([A-Za-z][A-Za-z0-9+/\-\s]{1,80}?)(?:\s+may\b|[:.;,])",
            cleaned,
            flags=re.IGNORECASE,
        )
        if not match:
            return ""
        candidate = match.group(1).strip(" :-;,")
        if ResponderAgent._looks_like_generic_label_ddi_target(candidate):
            return ""
        return candidate

    def _normalize_ddi_details(self, description: str, label: str) -> str:
        if description and description.lower() != "unclassified":
            return description
        label_text = self._humanize_ddi_label(label)
        if label_text:
            return label_text
        return description

    @staticmethod
    def _ddi_severity_priority(severity: str) -> int:
        normalized = str(severity or "").strip().lower()
        if normalized in {"contraindicated", "contraindication"}:
            return 3
        if normalized == "major":
            return 2
        if normalized == "moderate":
            return 1
        return 0

    @staticmethod
    def _ddi_label_priority(label: str) -> int:
        tokens = ResponderAgent._ddi_label_tokens(label)
        if "CI" in tokens:
            return 3
        if "C" in tokens:
            return 2
        if "P" in tokens:
            return 1
        return 0

    @staticmethod
    def _humanize_ddi_label(label: str) -> str:
        mapping = {
            "CI": "contraindication",
            "C": "caution",
            "P": "precaution",
        }
        parts: List[str] = []
        for token in ResponderAgent._ddi_label_tokens(label):
            mapped = mapping.get(token)
            if mapped and mapped not in parts:
                parts.append(mapped)
        return ", ".join(parts)

    @staticmethod
    def _ddi_label_tokens(label: str) -> List[str]:
        return [token.strip().upper() for token in str(label or "").split(",") if token.strip()]

    def _labeling_claim_for_item(self, item: Any) -> str:
        source_entity = str(item.metadata.get("source_entity", "")).strip()
        relationship = str(item.metadata.get("relationship", "")).strip().lower()
        target_entity = str(item.metadata.get("target_entity", "")).strip()
        snippet = self._clean_label_text(str(getattr(item, "snippet", "") or ""))

        if relationship == "indicated_for" and snippet:
            return f"{source_entity}: {snippet}"
        if relationship == "has_warning" and snippet:
            return f"{source_entity} warning: {snippet}"
        if relationship == "has_adverse_reaction" and snippet:
            return f"{source_entity} adverse reactions: {snippet}"
        if relationship == "has_contraindication" and snippet:
            return f"{source_entity} contraindications: {snippet}"
        if relationship == "use_in_special_population" and snippet:
            return f"{source_entity} special-population use: {snippet}"
        if relationship == "interacts_with":
            partner = self._ddi_partner_name(item)
            if source_entity and partner:
                return f"{source_entity} interaction information: {partner}"
            if snippet:
                return f"{source_entity} interaction information: {snippet}"
        if relationship == "has_dosing_guidance" and snippet:
            return f"{source_entity} dosing guidance: {snippet}"
        if relationship == "has_mechanism" and snippet:
            return f"{source_entity} mechanism: {snippet}"
        if relationship == "has_patient_drug_info" and target_entity:
            return f"{source_entity} has patient guidance: {target_entity}"
        if relationship == "has_official_label":
            if snippet:
                return f"{source_entity} official label summary: {snippet}"
            if target_entity:
                return f"{source_entity} official label available: {target_entity}"
        return ""

    def _repurposing_claim_for_item(self, item: Any) -> str:
        source_skill = str(getattr(item, "source_skill", "")).strip()
        source_entity = str(item.metadata.get("source_entity", "")).strip()
        relationship = str(item.metadata.get("relationship", "")).strip().lower()
        target_entity = str(item.metadata.get("target_entity", "")).strip()
        target_type = str(item.metadata.get("target_type", "")).strip().lower()
        snippet = self._clean_label_text(str(getattr(item, "snippet", "") or ""))
        phase = str(
            item.structured_payload.get("phase")
            or item.structured_payload.get("highest_phase")
            or ""
        ).strip()
        status = str(item.structured_payload.get("status") or "").strip()
        pmid = str(item.structured_payload.get("pmid") or "").strip()
        score = str(item.structured_payload.get("score") or "").strip()
        evidence = str(item.structured_payload.get("evidence") or "").strip()

        if relationship == "repurposing_evidence" and source_entity and target_entity:
            details = [detail for detail in (status, phase) if detail]
            suffix = f" ({'; '.join(details)})" if details else ""
            return f"{source_entity} has repurposing evidence for {target_entity}{suffix}"
        if source_skill == "RepurposeDrugs" and relationship == "repurposed_for" and source_entity and target_entity:
            details = [detail for detail in (status, phase) if detail]
            if score:
                details.append(f"score {score}")
            suffix = f" ({'; '.join(details)})" if details else ""
            return f"{source_entity} has exploratory repurposing signal for {target_entity}{suffix}"
        if relationship == "repurposed_for" and source_entity and target_entity:
            details = [detail for detail in (status, phase) if detail]
            if pmid:
                details.append(f"PMID {pmid}")
            if score:
                details.append(f"score {score}")
            suffix = f" ({'; '.join(details)})" if details else ""
            return f"{source_entity} has repurposing evidence for {target_entity}{suffix}"
        if (
            source_skill == "OREGANO"
            and relationship in {"clinical_signal", "repurposing_candidate"}
            and source_entity
            and target_entity
        ):
            details = [relationship.replace("_", " ")]
            if evidence:
                details.append(evidence)
            suffix = f" ({'; '.join(details)})" if details else ""
            return f"{source_entity} has exploratory repurposing signal for {target_entity}{suffix}"
        if relationship == "indicated_for" and source_entity and target_entity:
            if target_type == "disease":
                return f"{source_entity} is indicated for {target_entity}"
            if snippet:
                return f"{source_entity} label support: {snippet}"
        if relationship == "has_official_label":
            if snippet:
                return f"{source_entity} official label context: {snippet}"
            if target_entity:
                return f"{source_entity} official label available: {target_entity}"
        if relationship == "has_approved_entry":
            return f"{source_entity} has an approved-entry roster in DrugCentral"
        return ""

    def _summarize_repurposing_approved_indication_claims(
        self,
        primary_drug: str,
        items,
    ) -> List[ClaimSummary]:
        grouped: Dict[str, List[Any]] = defaultdict(list)
        for item in items:
            claim = self._repurposing_approved_indication_summary_claim(primary_drug, item)
            if not claim:
                continue
            grouped[claim].append(item)

        summaries: List[ClaimSummary] = []
        for claim, grouped_items in grouped.items():
            citations = [
                f"[{item.evidence_id}] {item.source_skill} ({item.source_locator})"
                for item in grouped_items
            ]
            summaries.append(
                ClaimSummary(
                    claim=claim,
                    confidence=score_claim_confidence(grouped_items),
                    evidence_ids=[item.evidence_id for item in grouped_items],
                    citations=citations,
                )
            )
        return summaries

    def _repurposing_approved_indication_summary_claim(
        self,
        primary_drug: str,
        item: Any,
    ) -> str:
        indication = self._extract_repurposing_approved_indication(item)
        if not indication:
            return ""
        drug_label = primary_drug.strip() if primary_drug else str(item.metadata.get("source_entity", "")).strip()
        if not drug_label:
            return ""
        return f"{drug_label} is approved for {indication}"

    def _extract_repurposing_approved_indication(self, item: Any) -> str:
        relationship = str(item.metadata.get("relationship", "")).strip().lower()
        if relationship != "indicated_for":
            return ""

        target_entity = str(item.metadata.get("target_entity", "")).strip()
        target_type = str(item.metadata.get("target_type", "")).strip().lower()
        if target_entity and target_type == "disease":
            return target_entity

        snippet = self._clean_label_text(str(getattr(item, "snippet", "") or ""))
        if not snippet:
            return ""

        patterns = (
            r"\bpatients?\s+with\s+([^.;]+?)(?:[.;]|$)",
            r"\bwith\s+([^.;]+?)(?:[.;]|$)",
            r"\bfor\s+(?:the treatment of\s+)?([^.;]+?)(?:[.;]|$)",
        )
        for pattern in patterns:
            match = re.search(pattern, snippet, flags=re.IGNORECASE)
            if not match:
                continue
            candidate = match.group(1).strip(" ,;:.")
            candidate = re.sub(r"\s+", " ", candidate).strip()
            if not candidate:
                continue
            candidate_lower = candidate.lower()
            if candidate_lower in {
                "indications and usage",
                "adults",
                "adult patients",
                "pediatric patients",
            }:
                continue
            return candidate_lower
        return ""

    def _mechanism_claim_for_item(self, item: Any, evidence_items=None) -> str:
        source_entity = str(item.metadata.get("source_entity", "")).strip()
        relationship = str(item.metadata.get("relationship", "")).strip().lower()
        target_entity = str(item.metadata.get("target_entity", "")).strip()
        snippet = self._clean_label_text(str(getattr(item, "snippet", "") or ""))
        mechanism = str(
            item.structured_payload.get("mechanism_of_action")
            or item.metadata.get("mechanism_of_action")
            or ""
        ).strip()
        action_phrase = self._mechanism_action_phrase(relationship)
        normalized_target = self._normalize_target_label(target_entity) or target_entity

        if self._is_mechanism_evidence_item(item):
            mechanism_text = mechanism
            if (
                mechanism_text
                and action_phrase
                and normalized_target
                and "fusion" in mechanism_text.lower()
                and self._has_independent_target_support(item, evidence_items)
            ):
                mechanism_text = f"{action_phrase} {normalized_target}"
            elif not mechanism_text and action_phrase and normalized_target:
                mechanism_text = f"{action_phrase} {normalized_target}"
            if not mechanism_text:
                mechanism_text = snippet or normalized_target
            if mechanism_text:
                return f"{source_entity} mechanism: {mechanism_text}"
        if self._is_target_evidence_item(item) and source_entity and target_entity:
            return f"{source_entity} targets {target_entity}."
        if "mechanism" in relationship and source_entity and target_entity:
            return f"{source_entity} mechanism involves {target_entity}"
        return ""

    def _has_independent_target_support(self, item: Any, evidence_items) -> bool:
        if not evidence_items:
            return False

        target_key = self._canonical_target_key(self._extract_target_label(item))
        current_source = str(getattr(item, "source_skill", "")).strip()
        if not target_key:
            return False

        for other in evidence_items:
            if other is item or not self._is_target_evidence_item(other):
                continue
            other_key = self._canonical_target_key(self._extract_target_label(other))
            if other_key != target_key:
                continue
            other_source = str(getattr(other, "source_skill", "")).strip()
            if other_source != current_source or not self._is_mechanism_evidence_item(other):
                return True
        return False

    @staticmethod
    def _mechanism_action_phrase(relationship: str) -> str:
        normalized = str(relationship or "").strip().lower()
        if "inhib" in normalized or "block" in normalized:
            return "inhibits"
        if "antagon" in normalized:
            return "antagonizes"
        if "agon" in normalized:
            return "agonizes"
        if "activat" in normalized:
            return "activates"
        if "modulat" in normalized:
            return "modulates"
        if "substr" in normalized:
            return "has substrate activity at"
        return ""

    def _pgx_claim_for_item(self, item: Any) -> str:
        source_entity = str(item.metadata.get("source_entity", "")).strip()
        target_entity = str(item.metadata.get("target_entity", "")).strip()
        relationship = str(item.metadata.get("relationship", "")).strip().lower()
        cpic_level = str(item.structured_payload.get("cpiclevel") or "").strip()
        clinpgx_level = str(item.structured_payload.get("clinpgxlevel") or "").strip()
        pgx_testing = str(item.structured_payload.get("pgxtesting") or "").strip()
        actionable = bool(item.structured_payload.get("usedforrecommendation"))

        if "guideline" in relationship:
            details: List[str] = []
            if cpic_level:
                details.append(f"CPIC level {cpic_level}")
            if clinpgx_level:
                details.append(f"ClinPGx {clinpgx_level}")
            if actionable or pgx_testing.lower().startswith("actionable"):
                details.append("actionable guidance")
            suffix = f" ({'; '.join(details)})" if details else ""
            return f"{source_entity} PGx guidance highlights {target_entity}{suffix}"
        if "pgx" in relationship or "association" in relationship:
            return f"{source_entity} has a pharmacogenomic association with {target_entity}"
        return ""

    def _adr_claim_for_item(self, item: Any) -> str:
        source_entity = str(item.metadata.get("source_entity", "")).strip()
        target_entity = str(item.metadata.get("target_entity", "")).strip()
        relationship = str(item.metadata.get("relationship", "")).strip().lower()
        if relationship in {
            "causes_adverse_event",
            "classified_adr",
            "has_side_effect",
            "associated_with_adverse_event",
        } and source_entity and target_entity:
            return f"{source_entity} serious safety signal: {target_entity}"
        return ""

    @staticmethod
    def _adr_item_priority(item: Any) -> tuple[int, int, float]:
        source_skill = str(getattr(item, "source_skill", "")).strip().lower()
        target_entity = str(item.metadata.get("target_entity", "")).strip().upper()
        priority = 0
        severe_terms = {
            "AGRANULOCYTOSIS": 5,
            "SEVERE NEUTROPENIA": 5,
            "NEUTROPENIA": 4,
            "GRANULOCYTOPENIA": 4,
            "LEUKOPENIA": 4,
            "MYOCARDITIS": 4,
            "CARDIOMYOPATHY": 4,
            "PERICARDITIS": 4,
            "SEIZURE": 4,
            "MYELOSUPPRESSION": 4,
            "CONSTIPATION": 2,
        }
        for term, score in severe_terms.items():
            if term in target_entity:
                priority = max(priority, score)
        source_priority = (
            2 if source_skill in {"adrecs", "sider", "nsides"}
            else 1 if source_skill in {"openfda human drug", "dailymed"}
            else 0
        )
        retrieval_score = float(getattr(item, "retrieval_score", 0.0) or 0.0)
        return (priority, source_priority, retrieval_score)

    @staticmethod
    def _looks_like_compound_identifier(value: str) -> bool:
        return bool(re.fullmatch(r"(?:cpd|dr):[A-Z0-9]+", value.strip(), re.IGNORECASE))

    @staticmethod
    def _extract_partner_from_text(text: str) -> str:
        if not text:
            return ""
        match = re.search(r"\bwith\s+([A-Za-z][A-Za-z0-9+/\-\s]{1,80})", text)
        if not match:
            return ""
        candidate = match.group(1)
        candidate = re.split(r"[.;,()]", candidate, maxsplit=1)[0].strip()
        if candidate.lower() in {"dr", "cpd"}:
            return ""
        return candidate

    @staticmethod
    def _clean_label_text(text: str) -> str:
        cleaned = re.sub(r"\s+", " ", text).strip()
        if not cleaned:
            return ""
        cleaned = re.sub(r"^\d+(?:\.\d+)?\s+[A-Z][A-Z\s/&-]{3,40}\s+", "", cleaned)
        cleaned = re.sub(r"^\d+(?:\.\d+)?\s+[A-Z][a-zA-Z\s/&-]{3,40}\s+", "", cleaned)
        for pattern, replacement in (
            (r"\bLactaton\b", "Lactation"),
            (r"\bdoage\b", "dosage"),
        ):
            cleaned = re.sub(pattern, replacement, cleaned, flags=re.IGNORECASE)
        return cleaned.strip()

    @staticmethod
    def _summarize_claims(
        evidence_items,
        assessments: List[ClaimAssessment],
    ) -> List[ClaimSummary]:
        grouped: Dict[str, List[Any]] = defaultdict(list)
        for item in evidence_items:
            grouped[item.claim].append(item)

        summaries: List[ClaimSummary] = []
        assessment_by_claim = {assessment.claim: assessment for assessment in assessments}
        for claim, items in grouped.items():
            assessment = assessment_by_claim.get(claim)
            confidence = (
                assessment.confidence if assessment is not None
                else score_claim_confidence(items)
            )
            citations = [
                f"[{item.evidence_id}] {item.source_skill} ({item.source_locator})"
                for item in items
            ]
            evidence_ids = (
                assessment.supporting_evidence_ids + assessment.contradicting_evidence_ids
                if assessment is not None
                else [item.evidence_id for item in items]
            )
            summaries.append(
                ClaimSummary(
                    claim=claim,
                    confidence=confidence,
                    evidence_ids=evidence_ids,
                    citations=citations,
                )
            )
        return sorted(summaries, key=lambda summary: summary.confidence, reverse=True)

    def _summarize_labeling_claims(
        self,
        evidence_items,
        assessments: List[ClaimAssessment],
    ) -> List[ClaimSummary]:
        grouped: Dict[str, List[Any]] = defaultdict(list)
        for item in evidence_items:
            grouped[item.claim].append(item)

        assessment_by_claim = {assessment.claim: assessment for assessment in assessments}
        ranked_records: List[tuple[ClaimSummary, str, tuple[int, int, float, float]]] = []
        for claim, items in grouped.items():
            assessment = assessment_by_claim.get(claim)
            confidence = (
                assessment.confidence if assessment is not None
                else score_claim_confidence(items)
            )
            citations = [
                f"[{item.evidence_id}] {item.source_skill} ({item.source_locator})"
                for item in items
            ]
            evidence_ids = (
                assessment.supporting_evidence_ids + assessment.contradicting_evidence_ids
                if assessment is not None
                else [item.evidence_id for item in items]
            )
            summary = ClaimSummary(
                claim=claim,
                confidence=confidence,
                evidence_ids=evidence_ids,
                citations=citations,
            )
            representative = max(items, key=self._labeling_item_priority)
            relationship = str(representative.metadata.get("relationship", "")).strip().lower()
            priority = (*self._labeling_item_priority(representative), confidence)
            ranked_records.append((summary, relationship, priority))

        ranked_records.sort(key=lambda record: record[2], reverse=True)
        diversified: List[ClaimSummary] = []
        overflow: List[ClaimSummary] = []
        seen_relationships = set()
        for summary, relationship, _ in ranked_records:
            if relationship and relationship not in seen_relationships:
                seen_relationships.add(relationship)
                diversified.append(summary)
            else:
                overflow.append(summary)
        return diversified + overflow

    def _summarize_target_claims(
        self,
        query: str,
        evidence_items,
        assessments: List[ClaimAssessment],
    ) -> List[ClaimSummary]:
        ranked_groups = self._rank_target_group_summaries(
            query,
            evidence_items,
            assessments,
        )
        if ranked_groups:
            return [group["summary"] for group in ranked_groups]

        return self._summarize_claims(evidence_items, assessments)

    def _rank_target_group_summaries(
        self,
        query: str,
        evidence_items,
        assessments: List[ClaimAssessment],
    ) -> List[Dict[str, Any]]:
        grouped: Dict[str, List[Any]] = defaultdict(list)
        for item in evidence_items:
            target_label = self._extract_target_label(item)
            if target_label:
                grouped[self._canonical_target_key(target_label)].append(item)

        if not grouped:
            return []

        drug_name = self._extract_primary_drug_name(query, evidence_items)
        assessment_by_claim = {assessment.claim: assessment for assessment in assessments}
        ranked_summaries: List[Dict[str, Any]] = []
        for _, items in grouped.items():
            label = self._choose_target_label(items)
            evidence_ids: List[str] = []
            citations: List[str] = []
            claim_confidences: List[float] = []
            retrieval_scores: List[float] = []
            potency_scores: List[float] = []
            specificity_scores: List[float] = []
            seen_ids = set()
            source_skills = set()
            relationship_bonus = 0
            for item in items:
                assessment = assessment_by_claim.get(item.claim)
                claim_confidences.append(
                    assessment.confidence if assessment is not None else score_claim_confidence([item])
                )
                retrieval_scores.append(float(getattr(item, "retrieval_score", 0.0) or 0.0))
                potency_scores.append(self._target_potency_score(item))
                specificity_scores.append(self._target_specificity_score(self._extract_target_label(item)))
                source_skills.add(str(getattr(item, "source_skill", "")).strip())
                relationship = str(item.metadata.get("relationship", "")).lower()
                if "target" in relationship or "bind" in relationship:
                    relationship_bonus = 1
                for evidence_id in (
                    assessment.supporting_evidence_ids + assessment.contradicting_evidence_ids
                    if assessment is not None
                    else [item.evidence_id]
                ):
                    if evidence_id not in seen_ids:
                        seen_ids.add(evidence_id)
                        evidence_ids.append(evidence_id)
                citation = f"[{item.evidence_id}] {item.source_skill} ({item.source_locator})"
                if citation not in citations:
                    citations.append(citation)
            summary = ClaimSummary(
                claim=f"{drug_name} targets {label}.",
                confidence=max(claim_confidences) if claim_confidences else 0.0,
                evidence_ids=evidence_ids,
                citations=citations,
            )
            ranking_key = (
                -len(source_skills),
                -len(evidence_ids),
                -relationship_bonus,
                -(max(specificity_scores) if specificity_scores else 0.0),
                -(max(potency_scores) if potency_scores else 0.0),
                -(max(claim_confidences) if claim_confidences else 0.0),
                -(max(retrieval_scores) if retrieval_scores else 0.0),
                label,
            )
            ranked_summaries.append(
                {
                    "summary": summary,
                    "label": label,
                    "canonical_key": self._canonical_target_key(label),
                    "items": list(items),
                    "source_skills": set(source_skills),
                    "evidence_count": len(evidence_ids),
                    "has_curated_source": bool(
                        source_skills.intersection(
                            {"BindingDB", "DrugBank", "DrugCentral", "IUPHAR", "TTD"}
                        )
                    ),
                    "ranking_key": ranking_key,
                }
            )
        return sorted(ranked_summaries, key=lambda item: item["ranking_key"])

    @staticmethod
    def _build_warnings(
        assessments: List[ClaimAssessment],
        claims: List[ClaimSummary],
        evidence_items,
    ) -> List[str]:
        warnings: List[str] = []
        for assessment in assessments:
            if assessment.verdict in {"uncertain", "contradicted"}:
                warnings.append(f"Evidence conflict detected for claim: {assessment.claim}")

        items_by_claim: Dict[str, List[Any]] = defaultdict(list)
        for item in evidence_items:
            items_by_claim[item.claim].append(item)

        if not warnings and len(claims) == 1 and claims[0].confidence < 0.45:
            warnings.append("Evidence is too sparse to support a confident conclusion.")
        return warnings

    @staticmethod
    def _build_limitations(
        assessments: List[ClaimAssessment],
        claims: List[ClaimSummary],
        evidence_items,
    ) -> List[str]:
        limitations: List[str] = []
        single_source_claims: List[str] = []
        predictive_only_claims: List[str] = []
        for assessment in assessments:
            limitations.extend(assessment.limitations)

        items_by_claim: Dict[str, List[Any]] = defaultdict(list)
        for item in evidence_items:
            items_by_claim[item.claim].append(item)

        for claim in claims:
            claim_items = items_by_claim[claim.claim]
            if not claim_items:
                continue
            unique_sources = {item.source_skill for item in claim_items}
            if len(unique_sources) == 1 and not any(
                claim.claim in limitation for limitation in limitations
            ):
                single_source_claims.append(claim.claim)
            if all(item.evidence_kind == "model_prediction" for item in claim_items):
                predictive_only_claims.append(claim.claim)

        if single_source_claims:
            limitations.extend(
                ResponderAgent._summarize_claim_limitations(
                    single_source_claims,
                    singular_prefix="Claim relies on a single source",
                    plural_prefix="Multiple claims rely on a single source",
                )
            )
        if predictive_only_claims:
            limitations.extend(
                ResponderAgent._summarize_claim_limitations(
                    predictive_only_claims,
                    singular_prefix="Claim is supported only by predictive evidence",
                    plural_prefix="Multiple claims are supported only by predictive evidence",
                )
            )
        return ResponderAgent._dedupe_preserve_order(limitations)

    @staticmethod
    def _build_citations(evidence_items) -> List[str]:
        citations = []
        seen = set()
        for item in evidence_items:
            citation = f"[{item.evidence_id}] {item.source_skill} — {item.source_locator}"
            if citation not in seen:
                seen.add(citation)
                citations.append(citation)
        return citations

    def _compute_task_outcome(
        self,
        query_type: str,
        evidence_items,
        claims: List[ClaimSummary],
        *,
        primary_drug: str = "",
    ) -> tuple[str, Dict[str, Any]]:
        normalized_task_type = normalize_task_type(query_type)

        if normalized_task_type == "direct_targets":
            sections, diagnostics = self._partition_direct_target_items(evidence_items)
            if sections["established_direct_targets"]:
                return "strong_answer", diagnostics
            if sections["association_only_signals"]:
                return "partial_with_weak_support", diagnostics
            return "honest_gap", diagnostics

        if normalized_task_type == "repurposing_evidence":
            sections, diagnostics = self._repurposing_answer_sections(primary_drug, evidence_items)
            if (
                sections["repurposing_evidence"]
                and diagnostics["strong_record_count"] > 0
                and diagnostics["strong_record_count"] == len(sections["repurposing_evidence"])
            ):
                return "strong_answer", diagnostics
            if sections["repurposing_evidence"]:
                return "partial_with_weak_support", diagnostics
            if evidence_items:
                return "partial_with_weak_support", diagnostics
            return "honest_gap", diagnostics

        if normalized_task_type == "mechanism_of_action":
            target_items = [item for item in evidence_items if self._is_target_evidence_item(item)]
            mechanism_items = [item for item in evidence_items if self._is_mechanism_evidence_item(item)]
            diagnostics = {
                "target_support_count": len(target_items),
                "mechanism_support_count": len(mechanism_items),
                "coverage_gap": not mechanism_items,
            }
            if target_items and mechanism_items:
                return "strong_answer", diagnostics
            if target_items or mechanism_items:
                return "partial_with_weak_support", diagnostics
            return "honest_gap", diagnostics

        if normalized_task_type == "pgx_guidance":
            strong_items = [item for item in evidence_items if self._is_strong_pgx_item(item)]
            diagnostics = {
                "strong_record_count": len(strong_items),
                "weak_support_count": max(0, len(evidence_items) - len(strong_items)),
            }
            if strong_items:
                return "strong_answer", diagnostics
            if evidence_items:
                return "partial_with_weak_support", diagnostics
            return "honest_gap", diagnostics

        diagnostics = {
            "evidence_count": len(evidence_items),
            "claim_count": len(claims),
        }
        if normalized_task_type == "major_adrs" and not claims:
            return "honest_gap", diagnostics
        if claims:
            return "partial_with_weak_support", diagnostics
        return "honest_gap", diagnostics

    @staticmethod
    def _is_target_evidence_item(item: Any) -> bool:
        relationship = str(item.metadata.get("relationship", "")).lower()
        target_entity = str(item.metadata.get("target_entity", "")).strip()
        target_type = str(item.metadata.get("target_type", "")).lower()
        claim_lower = str(getattr(item, "claim", "")).lower()

        if relationship in {"search_hit", "drug_lookup", "disease_lookup", "target_info"}:
            return False
        if any(noise in claim_lower for noise in ("search_hit", "unchecked", "no relevant target")):
            return False
        if target_type in {"cell_line", "disease", "drug_info", "disease_info", "unknown", "label_section"}:
            return False
        if target_entity and ResponderAgent._looks_like_cell_line(target_entity):
            return False
        return bool(
            relationship
            and any(
                marker in relationship
                for marker in (
                    "activity",
                    "target",
                    "bind",
                    "inhib",
                    "agon",
                    "antagon",
                    "substr",
                    "modulat",
                    "block",
                    "activat",
                )
            )
        )

    @staticmethod
    def _is_mechanism_evidence_item(item: Any) -> bool:
        relationship = str(item.metadata.get("relationship", "")).lower()
        structured_payload = getattr(item, "structured_payload", {}) or {}
        metadata = getattr(item, "metadata", {}) or {}
        target_entity = str(metadata.get("target_entity", "")).strip()
        target_type = str(metadata.get("target_type", "")).lower()
        has_action_relationship = bool(
            target_entity
            and target_type not in {"cell_line", "disease", "drug_info", "disease_info", "unknown", "label_section"}
            and any(
                marker in relationship
                for marker in (
                    "inhib",
                    "agon",
                    "antagon",
                    "substr",
                    "modulat",
                    "block",
                    "activat",
                )
            )
        )
        return bool(
            "mechanism" in relationship
            or structured_payload.get("mechanism_of_action")
            or metadata.get("mechanism_of_action")
            or has_action_relationship
        )

    @staticmethod
    def _is_strong_pgx_item(item: Any) -> bool:
        source_skill = str(getattr(item, "source_skill", "")).strip()
        relationship = str(item.metadata.get("relationship", "")).strip().lower()
        cpic_level = str(item.structured_payload.get("cpiclevel") or "").strip()
        clinpgx_level = str(item.structured_payload.get("clinpgxlevel") or "").strip()
        actionable = bool(item.structured_payload.get("usedforrecommendation"))
        return bool(
            source_skill in {"CPIC", "PharmGKB"}
            and (
                "guideline" in relationship
                or cpic_level
                or clinpgx_level
                or actionable
            )
        )

    def _partition_direct_target_items(self, evidence_items) -> tuple[Dict[str, List[Any]], Dict[str, Any]]:
        sections = {
            "established_direct_targets": [],
            "association_only_signals": [],
            "additional_context": [],
        }
        diagnostics: Dict[str, Any] = {
            "established_direct_target_count": 0,
            "association_only_signal_count": 0,
            "additional_context_count": 0,
        }

        for item in evidence_items:
            classification = classify_evidence_item("direct_targets", item)
            slot = classification.slot if classification.slot in sections else "additional_context"
            sections[slot].append(item)
            diagnostics[f"{slot[:-1] if slot.endswith('s') else slot}_count"] = diagnostics.get(
                f"{slot[:-1] if slot.endswith('s') else slot}_count",
                0,
            )

        diagnostics["established_direct_target_count"] = len(sections["established_direct_targets"])
        diagnostics["association_only_signal_count"] = len(sections["association_only_signals"])
        diagnostics["additional_context_count"] = len(sections["additional_context"])
        diagnostics["coverage_gap"] = len(sections["established_direct_targets"]) == 0
        return sections, diagnostics

    def _partition_repurposing_items(self, evidence_items) -> tuple[Dict[str, List[Any]], Dict[str, Any]]:
        sections = {
            "approved_indications": [],
            "repurposing_evidence": [],
            "supporting_signals": [],
        }
        diagnostics: Dict[str, Any] = {
            "strong_sources_attempted": ["RepoDB", "DrugCentral", "DrugBank", "DrugRepoBank"],
            "exploratory_repurposing_sources_attempted": ["RepurposeDrugs", "OREGANO"],
            "strong_record_count": 0,
            "strong_approved_indication_count": 0,
            "secondary_official_support_count": 0,
            "weak_support_count": 0,
        }

        for item in evidence_items:
            classification = classify_evidence_item("drug_repurposing", item)
            if classification.slot == "approved_indications":
                sections["approved_indications"].append(item)
            elif classification.slot == "repurposing_evidence":
                sections["repurposing_evidence"].append(item)
            else:
                sections["supporting_signals"].append(item)

            if (
                classification.tier == "strong_structured"
                and classification.slot == "repurposing_evidence"
            ):
                diagnostics["strong_record_count"] += 1
            elif classification.slot == "approved_indications" and classification.tier in {
                "strong_structured",
                "secondary_official_support",
            }:
                diagnostics["strong_approved_indication_count"] += 1
                if classification.tier == "secondary_official_support":
                    diagnostics["secondary_official_support_count"] += 1
            elif classification.tier == "secondary_official_support":
                diagnostics["secondary_official_support_count"] += 1
            else:
                diagnostics["weak_support_count"] += 1

        diagnostics["approved_indication_record_count"] = len(sections["approved_indications"])
        diagnostics["repurposing_record_count"] = len(sections["repurposing_evidence"])
        diagnostics["local_primary_missing"] = len(sections["repurposing_evidence"]) == 0
        diagnostics["online_weak_fallback_used"] = (
            diagnostics["secondary_official_support_count"] > 0
            or diagnostics["weak_support_count"] > 0
        )
        return sections, diagnostics

    def _repurposing_answer_sections(
        self,
        primary_drug: str,
        evidence_items,
    ) -> tuple[Dict[str, List[Any]], Dict[str, Any]]:
        sections, diagnostics = self._partition_repurposing_items(evidence_items)
        sections["approved_indications"] = self._filter_repurposing_approved_indication_items(
            primary_drug,
            sections["approved_indications"],
        )
        diagnostics["approved_indication_record_count"] = len(sections["approved_indications"])
        return sections, diagnostics

    @staticmethod
    def _subset_assessments_for_items(
        assessments: List[ClaimAssessment],
        items,
    ) -> List[ClaimAssessment]:
        allowed_claims = {item.claim for item in items}
        return [
            assessment
            for assessment in assessments
            if assessment.claim in allowed_claims
        ]

    def _mechanism_claim_groups(
        self,
        query: str,
        evidence_items,
        assessments: List[ClaimAssessment],
    ) -> tuple[List[Any], List[Any], List[ClaimSummary], List[ClaimSummary]]:
        target_items = [item for item in evidence_items if self._is_target_evidence_item(item)]
        mechanism_items = [item for item in evidence_items if self._is_mechanism_evidence_item(item)]
        target_claims = (
            self._summarize_target_claims(
                query,
                target_items,
                self._subset_assessments_for_items(assessments, target_items),
            )
            if target_items
            else []
        )
        mechanism_claims = (
            self._summarize_claims(
                mechanism_items,
                self._subset_assessments_for_items(assessments, mechanism_items),
            )
            if mechanism_items
            else []
        )
        return target_items, mechanism_items, target_claims, mechanism_claims

    def _prioritized_mechanism_claims(
        self,
        target_items,
        mechanism_claims: List[ClaimSummary],
    ) -> List[ClaimSummary]:
        if not mechanism_claims:
            return []

        direct_sections, _ = self._partition_direct_target_items(target_items)
        established_keys = {
            self._canonical_target_key(self._extract_target_label(item))
            for item in direct_sections["established_direct_targets"]
            if self._extract_target_label(item)
        }
        if not established_keys:
            return self._dedupe_claims_by_target_key(mechanism_claims)

        prioritized = [
            claim
            for claim in mechanism_claims
            if self._mechanism_claim_target_key(claim.claim) in established_keys
        ]
        return self._dedupe_claims_by_target_key(prioritized)

    def _dedupe_claims_by_target_key(
        self,
        claims: List[ClaimSummary],
    ) -> List[ClaimSummary]:
        deduped: List[ClaimSummary] = []
        seen_keys = set()
        for claim in claims:
            key = self._mechanism_claim_target_key(claim.claim) or claim.claim.strip().lower()
            if key in seen_keys:
                continue
            seen_keys.add(key)
            deduped.append(claim)
        return deduped

    def _primary_answer_fragments(
        self,
        task_type: str,
        claims: List[ClaimSummary],
    ) -> List[str]:
        normalized_task_type = normalize_task_type(task_type)
        fragments: List[str] = []
        seen_keys = set()
        for claim in claims:
            fragment = self._claim_to_target_fragment(claim.claim)
            key = (
                self._mechanism_claim_target_key(claim.claim)
                if normalized_task_type == "mechanism_of_action"
                else fragment.strip().lower()
            )
            if key in seen_keys:
                continue
            seen_keys.add(key)
            fragments.append(fragment)
        return fragments

    @staticmethod
    def _format_item_lines(items, empty_message: str, limit: int = 5) -> List[str]:
        seen = set()
        lines: List[str] = []
        for item in items:
            text = str(getattr(item, "claim", "") or getattr(item, "snippet", "")).strip()
            if not text:
                continue
            dedupe_key = text.lower()
            if dedupe_key in seen:
                continue
            seen.add(dedupe_key)
            if len(text) > 220:
                text = text[:217].rstrip() + "..."
            source_skill = str(getattr(item, "source_skill", "")).strip()
            evidence_id = str(getattr(item, "evidence_id", "")).strip()
            suffix_parts = [part for part in (source_skill, f"evidence {evidence_id}" if evidence_id else "") if part]
            suffix = f" ({'; '.join(suffix_parts)})" if suffix_parts else ""
            lines.append(f"- {text}{suffix}")
            if len(lines) >= limit:
                break
        return lines or [f"- {empty_message}"]

    @staticmethod
    def _format_claim_lines(
        claims: List[ClaimSummary],
        empty_message: str,
        limit: int = 5,
    ) -> List[str]:
        if not claims:
            return [f"- {empty_message}"]
        return [
            (
                f"- {claim.claim} "
                f"(confidence {claim.confidence:.2f}; evidence {', '.join(claim.evidence_ids[:4])})"
            )
            for claim in claims[:limit]
        ]

    def _build_direct_targets_section_lines(
        self,
        query: str,
        evidence_items,
        assessments: List[ClaimAssessment],
    ) -> List[str]:
        sections, _ = self._partition_direct_target_items(evidence_items)
        established_items = sections["established_direct_targets"]
        promotion_keys = {
            self._canonical_target_key(self._extract_target_label(item))
            for item in sections["association_only_signals"]
            if self._extract_target_label(item)
        }
        association_items = self._visible_association_only_items(
            evidence_items,
            established_items,
            sections["association_only_signals"],
        )
        ranked_established_groups = (
            self._rank_target_group_summaries(
                query,
                established_items,
                self._subset_assessments_for_items(assessments, established_items),
            )
            if established_items
            else []
        )
        established_claims: List[ClaimSummary] = []
        additional_direct_activity_groups: List[Dict[str, Any]] = []
        for group in ranked_established_groups:
            if group["has_curated_source"] or len(group["source_skills"]) >= 2:
                established_claims.append(group["summary"])
            else:
                additional_direct_activity_groups.append(group)

        additional_direct_activity_groups = sorted(
            additional_direct_activity_groups,
            key=lambda group: (
                0 if group["canonical_key"] in promotion_keys else 1,
                group["ranking_key"],
            ),
        )
        if len(established_claims) < 3 and additional_direct_activity_groups:
            promote_count = min(3 - len(established_claims), len(additional_direct_activity_groups))
            established_claims.extend(
                group["summary"] for group in additional_direct_activity_groups[:promote_count]
            )
            additional_direct_activity_groups = additional_direct_activity_groups[promote_count:]
        additional_direct_activity_claims = [
            group["summary"] for group in additional_direct_activity_groups
        ]

        association_claims = (
            self._summarize_target_claims(
                query,
                association_items,
                self._subset_assessments_for_items(assessments, association_items),
            )
            if association_items
            else []
        )

        lines = ["Established Direct Targets:"]
        lines.extend(
            self._format_claim_lines(
                established_claims,
                "No established direct-target evidence was retrieved.",
            )
        )
        if additional_direct_activity_claims:
            lines.extend(["", "Additional Direct Activity Hits:"])
            lines.extend(
                self._format_claim_lines(
                    additional_direct_activity_claims,
                    "No additional direct activity hits were retrieved.",
                )
            )
        lines.extend(["", "Association-Only Signals:"])
        lines.extend(
            self._format_claim_lines(
                association_claims,
                "No weaker association-only target signals were retrieved.",
            )
        )
        return lines

    def _visible_association_only_items(
        self,
        target_items,
        established_items,
        association_items,
    ) -> List[Any]:
        established_keys = {
            self._canonical_target_key(self._extract_target_label(item))
            for item in established_items
            if self._extract_target_label(item)
        }
        visible: List[Any] = []
        for item in association_items:
            target_label = self._extract_target_label(item)
            canonical_key = self._canonical_target_key(target_label)
            if canonical_key and canonical_key in established_keys:
                continue
            if self._is_low_value_open_targets_alias_signal(item, target_items):
                continue
            if self._is_fusion_component_only_signal(item, target_items):
                continue
            visible.append(item)
        return visible

    def _is_low_value_open_targets_alias_signal(self, item: Any, target_items) -> bool:
        source_skill = str(getattr(item, "source_skill", "")).strip()
        if source_skill != "Open Targets Platform":
            return False

        raw_label = self._extract_target_label(item)
        normalized_label = self._normalize_target_label(raw_label)
        if not raw_label or not normalized_label or raw_label.strip() == normalized_label:
            return False
        if not self._looks_like_target_symbol(normalized_label):
            return False

        lowered_label = raw_label.lower()
        if not any(marker in lowered_label for marker in ("activator of", "regulator of", "associated protein")):
            return False

        canonical_key = self._canonical_target_key(raw_label)
        for other in target_items:
            if other is item:
                continue
            other_key = self._canonical_target_key(self._extract_target_label(other))
            if other_key != canonical_key:
                continue
            if str(getattr(other, "source_skill", "")).strip() != "Open Targets Platform":
                return False
        return True

    def _is_fusion_component_only_signal(self, item: Any, target_items) -> bool:
        source_skill = str(getattr(item, "source_skill", "")).strip()
        if source_skill != "Open Targets Platform":
            return False

        mechanism_text = " ".join(
            part
            for part in (
                str(getattr(item, "snippet", "") or "").strip(),
                str((getattr(item, "structured_payload", {}) or {}).get("mechanism_of_action") or "").strip(),
                str((getattr(item, "metadata", {}) or {}).get("mechanism_of_action") or "").strip(),
            )
            if part
        ).strip()
        fusion_component_keys = self._fusion_component_target_keys(mechanism_text)
        if not fusion_component_keys:
            return False

        canonical_key = self._canonical_target_key(self._extract_target_label(item))
        if canonical_key not in fusion_component_keys:
            return False

        for other in target_items:
            if other is item:
                continue
            other_key = self._canonical_target_key(self._extract_target_label(other))
            if other_key != canonical_key:
                continue
            if str(getattr(other, "source_skill", "")).strip() != "Open Targets Platform":
                return False
        return True

    def _fusion_component_target_keys(self, mechanism_text: str) -> set[str]:
        lowered = str(mechanism_text or "").strip().lower()
        if "fusion" not in lowered:
            return set()

        prefix = lowered.split("fusion", 1)[0]
        tokens = re.findall(r"[a-z0-9-]{2,}", prefix)
        component_keys = {
            self._canonical_target_key(token)
            for token in tokens
            if token not in {"protein", "kinase", "receptor", "stem", "cell", "growth", "factor", "inhibitor"}
        }
        return {key for key in component_keys if key}

    def _build_mechanism_section_lines(
        self,
        query: str,
        evidence_items,
        assessments: List[ClaimAssessment],
        *,
        title_case: bool = False,
        include_targets: bool = True,
    ) -> List[str]:
        target_items, _, target_claims, mechanism_claims = self._mechanism_claim_groups(
            query,
            evidence_items,
            assessments,
        )
        mechanism_claims = self._prioritized_mechanism_claims(target_items, mechanism_claims)

        target_heading = "Targets Supported:" if title_case else "Targets supported:"
        mechanism_heading = "Mechanism Coverage:" if title_case else "Mechanism coverage:"

        lines: List[str] = []
        if include_targets:
            lines.append(target_heading)
            lines.extend(
                self._format_claim_lines(
                    target_claims,
                    "No target-support evidence was retrieved.",
                )
            )
            lines.extend(["", mechanism_heading])
        else:
            lines.append(mechanism_heading)
        lines.extend(
            self._format_claim_lines(
                mechanism_claims,
                "No direct mechanism-of-action evidence was retrieved.",
            )
        )
        if not mechanism_claims:
            lines.extend(
                [
                    "",
                    "Coverage gaps:",
                    "- Direct mechanism-of-action support was not retrieved, so this answer reflects target support rather than full MOA completeness.",
                ]
            )
        return lines

    def _render_direct_targets_answer(
        self,
        query: str,
        evidence_items,
        assessments: List[ClaimAssessment],
        warnings: List[str],
        limitations: List[str],
        web_section: tuple[str, List[str]],
    ) -> str:
        lines = [f"Query: {query}", "", "Known Targets:", ""]
        lines.extend(self._build_direct_targets_section_lines(query, evidence_items, assessments))

        web_heading, web_lines = web_section
        if web_lines:
            lines.extend(["", web_heading])
            lines.extend(web_lines)

        if warnings:
            lines.extend(["", "Warnings:"])
            lines.extend(f"- {warning}" for warning in warnings)

        if limitations:
            lines.extend(["", "Limitations:"])
            lines.extend(f"- {limitation}" for limitation in limitations)

        return "\n".join(lines)

    def _render_composite_answer(
        self,
        query: str,
        primary_drug: str,
        evidence_items,
        assessments: List[ClaimAssessment],
        primary_task_type: str,
        supporting_task_types: List[str],
        warnings: List[str],
        limitations: List[str],
        web_section: tuple[str, List[str]],
    ) -> str:
        lines = [f"Query: {query}", "", "Short Answer:"]

        primary_items = self._filter_items_for_task_type(
            primary_task_type,
            query=query,
            primary_drug=primary_drug,
            evidence_items=evidence_items,
        )
        primary_query_type = legacy_question_type_for_task_type(primary_task_type)
        if primary_query_type not in {"direct_targets", "target_profile"}:
            self._semanticize_claims_for_query_type(primary_query_type, primary_items)
        primary_assessments = self._subset_assessments_for_items(assessments, primary_items)
        if not primary_assessments and primary_items:
            primary_assessments = assess_claims(primary_items)
        primary_claims = self._composite_primary_claims(
            query,
            primary_drug,
            primary_task_type,
            primary_items,
            primary_assessments,
        )
        if primary_claims:
            primary_fragments = self._primary_answer_fragments(primary_task_type, primary_claims)
            primary_labels = ", ".join(primary_fragments[:3])
            lines.append(f"- Primary supported answer: {primary_labels}")
        else:
            lines.append("- Structured evidence is available, but the primary answer remains incomplete.")

        task_order = [primary_task_type] + list(supporting_task_types)
        for task_type in task_order:
            task_items = self._filter_items_for_task_type(
                task_type,
                query=query,
                primary_drug=primary_drug,
                evidence_items=evidence_items,
            )
            task_query_type = legacy_question_type_for_task_type(task_type)
            if task_type == "labeling_summary" and primary_task_type == "repurposing_evidence":
                task_query_type = "drug_repurposing"
            if task_query_type not in {"direct_targets", "target_profile"}:
                self._semanticize_claims_for_query_type(task_query_type, task_items)
            task_assessments = (
                primary_assessments
                if task_type == primary_task_type
                else self._subset_assessments_for_items(assessments, task_items)
            )
            if not task_assessments and task_items:
                task_assessments = assess_claims(task_items)
            if task_type != primary_task_type and not task_items:
                continue

            lines.append("")
            if task_type == "direct_targets":
                lines.extend(self._build_direct_targets_section_lines(query, task_items, task_assessments))
            elif task_type == "mechanism_of_action":
                lines.extend(
                    self._build_mechanism_section_lines(
                        query,
                        task_items,
                        task_assessments,
                        title_case=True,
                        include_targets=False,
                    )
                )
            elif task_type == "repurposing_evidence":
                sections, _ = self._repurposing_answer_sections(primary_drug, task_items)
                repurposing_items = sections["repurposing_evidence"]
                repurposing_claims = self._summarize_claims(
                    repurposing_items,
                    self._subset_assessments_for_items(task_assessments, repurposing_items),
                )
                lines.append("Repurposing Evidence:")
                lines.extend(
                    self._format_claim_lines(
                        repurposing_claims,
                        "No repurposing evidence was retrieved.",
                    )
                )
            elif task_type == "labeling_summary" and primary_task_type == "repurposing_evidence":
                sections, _ = self._repurposing_answer_sections(primary_drug, task_items)
                lines.append("Labeling Summary:")
                lines.extend(
                    self._format_item_lines(
                        sections["approved_indications"],
                        "No approved-indication evidence was retrieved.",
                    )
                )
            else:
                task_claims = (
                    self._summarize_labeling_claims(task_items, task_assessments)
                    if task_type == "labeling_summary"
                    else self._summarize_claims(task_items, task_assessments)
                )
                lines.append(f"{task_type.replace('_', ' ').title()}:")
                lines.extend(
                    self._format_claim_lines(
                        task_claims,
                        "No structured evidence was retrieved for this section.",
                    )
                )

        web_heading, web_lines = web_section
        if web_lines:
            lines.extend(["", web_heading])
            lines.extend(web_lines)

        if warnings:
            lines.extend(["", "Warnings:"])
            lines.extend(f"- {warning}" for warning in warnings)

        if limitations:
            lines.extend(["", "Limitations:"])
            lines.extend(f"- {limitation}" for limitation in limitations)

        return "\n".join(lines)

    def _render_repurposing_answer(
        self,
        query: str,
        primary_drug: str,
        evidence_items,
        warnings: List[str],
        limitations: List[str],
        web_section: tuple[str, List[str]],
    ) -> str:
        sections, diagnostics = self._repurposing_answer_sections(primary_drug, evidence_items)
        approved_claims = self._summarize_repurposing_approved_indication_claims(
            primary_drug,
            sections["approved_indications"],
        )
        lines = [f"Query: {query}", "", "Approved indications:"]
        lines.extend(
            self._format_claim_lines(
                approved_claims,
                "No approved-indication evidence was retrieved.",
            )
        )

        lines.extend(["", "Repurposing evidence:"])
        lines.extend(
            self._format_item_lines(
                sections["repurposing_evidence"],
                "No strong repurposing evidence was retrieved.",
            )
        )

        lines.extend(["", "Supporting signals:"])
        lines.extend(
            self._format_item_lines(
                sections["supporting_signals"],
                "No additional supporting signals were retrieved.",
            )
        )

        coverage_gaps: List[str] = []
        if not sections["repurposing_evidence"]:
            coverage_gaps.append(
                "Curated repurposing rows were not retrieved, so the answer cannot claim strong repurposing support."
            )
        elif diagnostics["strong_record_count"] == 0:
            coverage_gaps.append(
                "Retrieved repurposing signals were exploratory or lower-confidence, so they should not be treated as established repurposing evidence."
            )
        if not sections["approved_indications"]:
            coverage_gaps.append(
                "Approved-indication coverage is incomplete for this repurposing-style question."
            )
        if coverage_gaps:
            lines.extend(["", "Coverage gaps:"])
            lines.extend(f"- {gap}" for gap in coverage_gaps)

        web_heading, web_lines = web_section
        if web_lines:
            lines.extend(["", web_heading])
            lines.extend(web_lines)

        if warnings:
            lines.extend(["", "Warnings:"])
            lines.extend(f"- {warning}" for warning in warnings)

        if limitations:
            lines.extend(["", "Limitations:"])
            lines.extend(f"- {limitation}" for limitation in limitations)

        return "\n".join(lines)

    def _filter_repurposing_approved_indication_items(
        self,
        primary_drug: str,
        items,
    ) -> List[Any]:
        return self._dedupe_repurposing_approved_indication_items(primary_drug, items)

    def _render_mechanism_answer(
        self,
        query: str,
        evidence_items,
        assessments: List[ClaimAssessment],
        warnings: List[str],
        limitations: List[str],
        web_section: tuple[str, List[str]],
    ) -> str:
        lines = [f"Query: {query}", ""]
        lines.extend(
            self._build_mechanism_section_lines(
                query,
                evidence_items,
                assessments,
                title_case=False,
            )
        )
        web_heading, web_lines = web_section
        if web_lines:
            lines.extend(["", web_heading])
            lines.extend(web_lines)

        if warnings:
            lines.extend(["", "Warnings:"])
            lines.extend(f"- {warning}" for warning in warnings)

        if limitations:
            lines.extend(["", "Limitations:"])
            lines.extend(f"- {limitation}" for limitation in limitations)

        return "\n".join(lines)

    @staticmethod
    def _render_answer_text(
        query: str,
        query_type: str,
        claims: List[ClaimSummary],
        warnings: List[str],
        limitations: List[str],
        web_section: tuple[str, List[str]],
    ) -> str:
        if not claims:
            return (
                f"Query: {query}\n\n"
                "Insufficient evidence found to answer the query."
            )

        lines = [f"Query: {query}", "", f"{ResponderAgent._claim_section_heading(query_type)}:"]
        for claim in claims[:5]:
            lines.append(
                f"- {claim.claim} "
                f"(confidence {claim.confidence:.2f}; evidence {', '.join(claim.evidence_ids)})"
            )

        if query_type == "ddi_mechanism" and (
            len(claims) <= 1
            or any("single source" in limitation.lower() for limitation in limitations)
        ):
            lines.extend(
                [
                    "",
                    "Mechanism coverage:",
                    "- Retrieved mechanistic interaction evidence is sparse and does not establish complete interaction-mechanism coverage.",
                ]
            )

        web_heading, web_lines = web_section

        if web_lines:
            lines.extend(["", web_heading])
            lines.extend(web_lines)

        if warnings:
            lines.extend(["", "Warnings:"])
            lines.extend(f"- {warning}" for warning in warnings)

        if limitations:
            lines.extend(["", "Limitations:"])
            lines.extend(f"- {limitation}" for limitation in limitations)

        return "\n".join(lines)

    @staticmethod
    def _render_target_answer(
        query: str,
        claims: List[ClaimSummary],
        warnings: List[str],
        limitations: List[str],
        web_section: tuple[str, List[str]],
    ) -> str:
        if not claims:
            return (
                f"Query: {query}\n\n"
                "Insufficient evidence found to answer the query."
            )

        lines = [f"Query: {query}", "", "Known Targets:"]
        display_claims = claims[:5]
        for claim in display_claims:
            target_label = claim.claim.replace(" targets ", " -> ").rstrip(".")
            lines.append(
                f"- {target_label} "
                f"(confidence {claim.confidence:.2f}; evidence {', '.join(claim.evidence_ids[:4])})"
            )

        if len(claims) > len(display_claims):
            lines.extend(
                [
                    "",
                    "Additional target-like activity evidence is available in the Evidence Summary.",
                ]
            )

        if warnings:
            lines.extend(["", "Warnings:"])
            lines.extend(f"- {warning}" for warning in warnings[:5])

        web_heading, web_lines = web_section

        if web_lines:
            lines.extend(["", web_heading])
            lines.extend(web_lines)

        if limitations:
            lines.extend(["", "Limitations:"])
            lines.extend(f"- {limitation}" for limitation in limitations[:8])

        return "\n".join(lines)

    @staticmethod
    def _claim_section_heading(query_type: str) -> str:
        return {
            "ddi": "Clinically Important Interactions",
            "ddi_mechanism": "Clinically Important Interactions",
            "labeling": "Structured Labeling Findings",
            "pharmacogenomics": "Structured PGx Findings",
            "adr": "Structured Safety Findings",
            "drug_repurposing": "Structured Repurposing Evidence",
            "mechanism": "Mechanistic Findings",
        }.get(str(query_type or "").strip().lower(), "Key Claims")

    @staticmethod
    def _claim_to_target_fragment(claim: str) -> str:
        text = str(claim or "").strip().rstrip(".")
        if " interacts with " in text:
            return text
        if " targets " in text:
            return text.split(" targets ", 1)[1].strip()
        if ": " in text:
            return text.split(": ", 1)[1].strip()
        return text

    @classmethod
    def _mechanism_claim_target_key(cls, claim: str) -> str:
        fragment = cls._claim_to_target_fragment(claim)
        normalized = re.sub(
            r"^(?:inhibits|agonizes|antagonizes|activates|modulates|targets|has substrate activity at|mechanism involves)\s+",
            "",
            fragment,
            flags=re.IGNORECASE,
        ).strip()
        normalized = normalized.split("(", 1)[0].strip()
        return cls._canonical_target_key(normalized or fragment)

    @staticmethod
    def _extract_target_label(item: Any) -> str:
        target_label = str(item.metadata.get("target_entity", "")).strip()
        if target_label:
            return target_label
        claim = item.claim.strip().rstrip(".")
        for token in (" targets ", " linked_target ", " has_ic50_activity ", " has_ki_activity ", " has_kd_activity ", " has_ec50_activity "):
            if token in claim:
                return claim.split(token, 1)[1].strip()
        return ""

    @staticmethod
    def _choose_target_label(items: List[Any]) -> str:
        labels = [ResponderAgent._normalize_target_label(ResponderAgent._extract_target_label(item)) for item in items]
        labels = [label for label in labels if label]
        if not labels:
            return "unknown target"
        symbol_like = [label for label in labels if ResponderAgent._looks_like_target_symbol(label)]
        if symbol_like:
            return min(symbol_like, key=len)
        return max(labels, key=len)

    @staticmethod
    def _canonical_target_key(label: str) -> str:
        normalized = ResponderAgent._normalize_target_label(label)
        if normalized:
            return normalized.lower()
        cleaned = re.sub(r"[^a-z0-9]+", " ", label.lower()).strip()
        return cleaned

    @staticmethod
    def _normalize_target_label(label: str) -> str:
        cleaned = re.sub(r"\s+", " ", str(label).strip())
        if not cleaned:
            return ""

        normalized = cleaned.lower()
        alias_map = {
            "tyrosine-protein kinase abl1": "ABL1",
            "tyrosine-protein kinase abl": "ABL1",
            "abl proto-oncogene 1, non-receptor tyrosine kinase": "ABL1",
            "abl1": "ABL1",
            "abl": "ABL1",
            "mast/stem cell growth factor receptor kit": "KIT",
            "kit proto-oncogene, receptor tyrosine kinase": "KIT",
            "kit": "KIT",
            "bcr activator of rhogef and gtpase": "BCR",
            "bcr": "BCR",
            "platelet-derived growth factor receptor beta": "PDGFRB",
            "platelet derived growth factor receptor beta": "PDGFRB",
            "pdgfrb": "PDGFRB",
            "platelet-derived growth factor receptor alpha": "PDGFRA",
            "pdgfra": "PDGFRA",
            "receptor-type tyrosine-protein kinase flt3": "FLT3",
            "flt3": "FLT3",
            "macrophage colony-stimulating factor 1 receptor": "CSF1R",
            "csf1r": "CSF1R",
            "epidermal growth factor receptor": "EGFR",
            "egfr": "EGFR",
            "proto-oncogene tyrosine-protein kinase src": "SRC",
            "src": "SRC",
        }
        if normalized in alias_map:
            return alias_map[normalized]

        token_match = re.search(r"\b([A-Z0-9-]{2,8})\b$", cleaned.upper())
        if token_match:
            token = token_match.group(1)
            if (
                token[0].isalpha()
                and token not in {"TYPE", "ALPHA", "BETA", "GAMMA", "RECEPTOR", "KINASE", "PROTEIN", "GTPASE"}
                and not token.startswith(("ALPHA", "BETA", "GAMMA", "DELTA", "EPSILON"))
                and (any(ch.isdigit() for ch in token) or len(token) <= 4)
            ):
                return token
        return cleaned

    @staticmethod
    def _looks_like_target_symbol(label: str) -> bool:
        return bool(re.fullmatch(r"[A-Z0-9-]{2,8}", label.strip()))

    @staticmethod
    def _target_specificity_score(label: str) -> float:
        normalized = ResponderAgent._normalize_target_label(label)
        if not normalized:
            return 0.0
        if ResponderAgent._looks_like_target_symbol(normalized):
            return 1.0

        lowered = normalized.lower()
        generic_family_labels = {
            "platelet-derived growth factor receptor",
            "tyrosine-protein kinase",
            "protein kinase",
            "receptor",
        }
        if lowered in generic_family_labels:
            return 0.1
        return 0.4

    @staticmethod
    def _target_potency_score(item: Any) -> float:
        value = None
        structured_payload = getattr(item, "structured_payload", {}) or {}
        for key in ("affinity_value", "value", "standard_value"):
            raw = structured_payload.get(key)
            parsed = ResponderAgent._coerce_float(raw)
            if parsed is not None and parsed > 0:
                value = parsed
                break

        if value is None:
            snippet = str(getattr(item, "snippet", "") or "")
            match = re.search(r"\b(?:IC50|Ki|Kd|EC50)\s*=\s*([0-9]+(?:\.[0-9]+)?)", snippet, re.IGNORECASE)
            if match:
                value = ResponderAgent._coerce_float(match.group(1))

        if value is None or value <= 0:
            return 0.0

        # Smaller potency values are generally stronger evidence for a direct
        # target-like interaction. Compress to [0, 1] to avoid dominating
        # source-count and support-count signals.
        return max(0.0, 1.0 - min(log10(max(value, 1.0)), 6.0) / 6.0)

    @staticmethod
    def _coerce_float(value: Any) -> float | None:
        try:
            if value is None or value == "":
                return None
            return float(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _looks_like_cell_line(label: str) -> bool:
        normalized = label.strip().upper().replace("-", "")
        return bool(re.fullmatch(r"[A-Z]{1,5}\d{2,}", normalized))

    @staticmethod
    def _extract_primary_drug_name(query: str, evidence_items) -> str:
        for item in evidence_items:
            source_entity = str(item.metadata.get("source_entity", "")).strip()
            if source_entity:
                return source_entity
            claim = str(getattr(item, "claim", "")).strip()
            if " targets " in claim:
                return claim.split(" targets ", 1)[0].strip() or "This drug"
        lowered = (query or "").lower()
        match = re.search(r"targets?\s+of\s+([a-z0-9\-]+)", lowered)
        if match:
            return match.group(1)
        match = re.search(r"does\s+([a-z0-9\-]+)\s+target", lowered)
        if match:
            return match.group(1)
        return "This drug"

    @staticmethod
    def _dedupe_preserve_order(lines: List[str]) -> List[str]:
        deduped: List[str] = []
        seen = set()
        for line in lines:
            normalized = line.strip()
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            deduped.append(normalized)
        return deduped

    @staticmethod
    def _summarize_claim_limitations(
        claims: List[str],
        *,
        singular_prefix: str,
        plural_prefix: str,
        max_examples: int = 3,
    ) -> List[str]:
        normalized_claims = ResponderAgent._dedupe_preserve_order(claims)
        if not normalized_claims:
            return []
        if len(normalized_claims) == 1:
            return [f"{singular_prefix}."]

        examples = normalized_claims[:max_examples]
        summary = f"{plural_prefix} ({len(normalized_claims)} claims)."
        if examples:
            summary += f" Examples: {'; '.join(examples)}."
        return [summary]
