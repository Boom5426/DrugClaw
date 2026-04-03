"""
Responder Agent - Generates intermediate answers based on current evidence
"""
import re
from collections import defaultdict
from math import log10
from typing import List, Dict, Any

from .claim_assessment import ClaimAssessment, assess_claims
from .evidence import ClaimSummary, FinalAnswer, score_answer_confidence, score_claim_confidence
from .models import AgentState, EvidencePath
from .llm_client import LLMClient
from .query_plan import infer_entities_from_query, infer_question_type_from_query

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
            if error:
                diagnostic_lines.append(f"- {skill}: {error}")
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
        )

    def _respond_from_evidence(self, state: AgentState) -> None:
        final_answer = self._build_final_answer(
            state.original_query,
            state.evidence_items,
            claim_assessments=state.claim_assessments,
        )
        state.final_answer_structured = final_answer
        state.current_answer = final_answer.answer_text
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
    ) -> FinalAnswer:
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
            )

        filtered_items = list(evidence_items)
        query_type = infer_question_type_from_query(query)
        if self._is_target_lookup_query(query):
            filtered_items = self._filter_target_evidence_items(filtered_items) or filtered_items
        elif query_type in {"ddi", "ddi_mechanism"}:
            filtered_items = self._filter_ddi_evidence_items(filtered_items) or filtered_items
        elif query_type == "pharmacogenomics":
            filtered_items = self._filter_pgx_evidence_items(query, filtered_items) or filtered_items
        elif query_type == "adr":
            filtered_items = self._filter_adr_evidence_items(filtered_items) or filtered_items
        elif query_type == "labeling":
            filtered_items = self._filter_labeling_evidence_items(query, filtered_items) or filtered_items
        if not self._is_target_lookup_query(query):
            self._semanticize_claims_for_query_type(query_type, filtered_items)

        assessments = list(claim_assessments or [])
        if not assessments:
            assessments = assess_claims(filtered_items)
        else:
            allowed_claims = {item.claim for item in filtered_items}
            assessments = [
                assessment for assessment in assessments
                if assessment.claim in allowed_claims
            ]

        claims = (
            self._summarize_target_claims(query, filtered_items, assessments)
            if self._is_target_lookup_query(query)
            else self._summarize_claims(filtered_items, assessments)
        )
        warnings = self._build_warnings(assessments, claims, filtered_items)
        limitations = self._build_limitations(assessments, claims, filtered_items)
        citations = self._build_citations(filtered_items)
        answer_text = (
            self._render_target_answer(query, claims, warnings, limitations)
            if self._is_target_lookup_query(query)
            else self._render_answer_text(query, claims, warnings, limitations)
        )

        return FinalAnswer(
            answer_text=answer_text,
            summary_confidence=score_answer_confidence(claims),
            key_claims=claims,
            evidence_items=list(filtered_items),
            citations=citations,
            limitations=limitations,
            warnings=warnings,
        )

    @staticmethod
    def _is_target_lookup_query(query: str) -> bool:
        lowered = (query or "").lower()
        return "target" in lowered

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

    def _filter_ddi_evidence_items(self, evidence_items) -> List[Any]:
        ddi_like_items: List[Any] = []
        informative_items: List[Any] = []

        for item in evidence_items:
            relationship = str(item.metadata.get("relationship", "")).lower()
            source_skill = str(getattr(item, "source_skill", "")).strip().lower()
            if (
                "interaction" not in relationship
                and "ddi" not in relationship
                and source_skill not in {"ddinter", "kegg drug", "mecddi", "drugbank"}
            ):
                continue

            ddi_like_items.append(item)
            description = str(
                item.structured_payload.get("ddi_description")
                or item.structured_payload.get("description")
                or ""
            ).strip()
            target_entity = str(item.metadata.get("target_entity", "")).strip()
            partner = ""
            if target_entity and not self._looks_like_compound_identifier(target_entity):
                partner = target_entity
            if not partner:
                partner = self._extract_partner_from_text(description) or self._extract_partner_from_text(
                    str(getattr(item, "snippet", "") or "")
                )
            if partner and self._looks_like_compound_identifier(partner):
                partner = ""

            if partner or (description and description.lower() != "unclassified"):
                informative_items.append(item)

        if informative_items:
            return informative_items
        if ddi_like_items:
            return ddi_like_items
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
        noise_targets = {
            "SCHIZOPHRENIA",
            "OFF LABEL USE",
            "TREATMENT NONCOMPLIANCE",
            "PSYCHOTIC DISORDER",
            "HOSPITALISATION",
            "DRUG INEFFECTIVE",
            "DRUG INTERACTION",
            "TOXICITY TO VARIOUS AGENTS",
        }
        filtered = []
        for item in evidence_items:
            relationship = str(item.metadata.get("relationship", "")).lower()
            target_entity = str(item.metadata.get("target_entity", "")).strip().upper()
            if relationship != "causes_adverse_event":
                continue
            if target_entity in noise_targets:
                continue
            filtered.append(item)
        return filtered

    def _filter_labeling_evidence_items(self, query: str, evidence_items) -> List[Any]:
        primary_drug = self._extract_query_drug_name(query)
        filtered = list(evidence_items)

        if primary_drug:
            strict_matches = []
            for item in filtered:
                source_entity = str(item.metadata.get("source_entity", "")).strip().lower()
                if not source_entity:
                    continue
                if primary_drug not in source_entity:
                    continue
                if " and " in source_entity and not source_entity.startswith(primary_drug):
                    continue
                strict_matches.append(item)
            if strict_matches:
                filtered = strict_matches

        primary_label_relationships = {
            "indicated_for",
            "has_warning",
            "has_adverse_reaction",
            "interacts_with",
            "has_mechanism",
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

        return filtered

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

    def _semanticize_claims_for_query_type(self, query_type: str, evidence_items) -> None:
        for item in evidence_items:
            semantic_claim = self._semantic_claim_for_item(query_type, item)
            if semantic_claim:
                item.claim = semantic_claim

    def _semantic_claim_for_item(self, query_type: str, item: Any) -> str:
        if query_type in {"ddi", "ddi_mechanism"}:
            return self._ddi_claim_for_item(item)
        if query_type == "labeling":
            return self._labeling_claim_for_item(item)
        if query_type == "pharmacogenomics":
            return self._pgx_claim_for_item(item)
        if query_type == "adr":
            return self._adr_claim_for_item(item)
        return ""

    def _ddi_claim_for_item(self, item: Any) -> str:
        source_entity = str(item.metadata.get("source_entity", "")).strip()
        target_entity = str(item.metadata.get("target_entity", "")).strip()
        description = str(
            item.structured_payload.get("ddi_description")
            or item.structured_payload.get("description")
            or ""
        ).strip()
        label = str(item.structured_payload.get("ddi_label") or "").strip()
        snippet = str(getattr(item, "snippet", "") or "").strip()

        partner = ""
        if target_entity and not self._looks_like_compound_identifier(target_entity):
            partner = target_entity
        if not partner:
            partner = self._extract_partner_from_text(description) or self._extract_partner_from_text(snippet)
        if partner and self._looks_like_compound_identifier(partner):
            partner = ""

        if description.lower().startswith("enzyme:"):
            enzyme = description.split(":", 1)[1].strip()
            if enzyme:
                return f"{source_entity} interaction mechanism involves {enzyme}"
        if source_entity and partner:
            details = description or label
            if details:
                return f"{source_entity} interacts with {partner} ({details})"
            return f"{source_entity} interacts with {partner}"
        if source_entity and description.lower() == "unclassified":
            return f"{source_entity} has unresolved KEGG interaction entries"
        if source_entity and description:
            return f"{source_entity} has a clinically important interaction: {description}"
        return ""

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
        if relationship == "interacts_with" and snippet:
            return f"{source_entity} interaction information: {snippet}"
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
        if relationship == "causes_adverse_event" and source_entity and target_entity:
            return f"{source_entity} serious safety signal: {target_entity}"
        return ""

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

    def _summarize_target_claims(
        self,
        query: str,
        evidence_items,
        assessments: List[ClaimAssessment],
    ) -> List[ClaimSummary]:
        grouped: Dict[str, List[Any]] = defaultdict(list)
        for item in evidence_items:
            target_label = self._extract_target_label(item)
            if target_label:
                grouped[self._canonical_target_key(target_label)].append(item)

        if not grouped:
            return self._summarize_claims(evidence_items, assessments)

        drug_name = self._extract_primary_drug_name(query, evidence_items)
        assessment_by_claim = {assessment.claim: assessment for assessment in assessments}
        ranked_summaries: List[tuple[tuple[Any, ...], ClaimSummary]] = []
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
            ranked_summaries.append(
                (
                    (
                        len(source_skills),
                        len(evidence_ids),
                        relationship_bonus,
                        max(specificity_scores) if specificity_scores else 0.0,
                        max(potency_scores) if potency_scores else 0.0,
                        max(claim_confidences) if claim_confidences else 0.0,
                        max(retrieval_scores) if retrieval_scores else 0.0,
                        label,
                    ),
                    summary,
                )
            )
        return [
            summary
            for _, summary in sorted(ranked_summaries, key=lambda item: item[0], reverse=True)
        ]

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

    @staticmethod
    def _render_answer_text(
        query: str,
        claims: List[ClaimSummary],
        warnings: List[str],
        limitations: List[str],
    ) -> str:
        if not claims:
            return (
                f"Query: {query}\n\n"
                "Insufficient evidence found to answer the query."
            )

        lines = [f"Query: {query}", "", "Key Claims:"]
        for claim in claims[:5]:
            lines.append(
                f"- {claim.claim} "
                f"(confidence {claim.confidence:.2f}; evidence {', '.join(claim.evidence_ids)})"
            )

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

        if limitations:
            lines.extend(["", "Limitations:"])
            lines.extend(f"- {limitation}" for limitation in limitations[:8])

        return "\n".join(lines)

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
            if token not in {"TYPE", "ALPHA", "BETA", "GAMMA", "RECEPTOR", "KINASE", "PROTEIN", "GTPASE"}:
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
            return [f"{singular_prefix}: {normalized_claims[0]}"]

        examples = normalized_claims[:max_examples]
        summary = f"{plural_prefix} ({len(normalized_claims)} claims)."
        if examples:
            summary += f" Examples: {'; '.join(examples)}."
        return [summary]
