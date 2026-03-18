"""
Responder Agent - Generates intermediate answers based on current evidence
"""
from collections import defaultdict
from typing import List, Dict, Any

from .claim_assessment import ClaimAssessment, assess_claims
from .evidence import ClaimSummary, FinalAnswer, score_answer_confidence, score_claim_confidence
from .models import AgentState, EvidencePath
from .llm_client import LLMClient

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

    def _respond_from_evidence(self, state: AgentState) -> None:
        final_answer = self._build_final_answer(
            state.original_query,
            state.evidence_items,
            claim_assessments=state.claim_assessments,
        )
        state.final_answer_structured = final_answer
        state.current_answer = final_answer.answer_text

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

        assessments = list(claim_assessments or [])
        if not assessments:
            assessments = assess_claims(evidence_items)

        claims = self._summarize_claims(evidence_items, assessments)
        warnings = self._build_warnings(assessments, claims, evidence_items)
        limitations = self._build_limitations(assessments, claims, evidence_items)
        citations = self._build_citations(evidence_items)
        answer_text = self._render_answer_text(query, claims, warnings, limitations)

        return FinalAnswer(
            answer_text=answer_text,
            summary_confidence=score_answer_confidence(claims),
            key_claims=claims,
            evidence_items=list(evidence_items),
            citations=citations,
            limitations=limitations,
            warnings=warnings,
        )

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
        for assessment in assessments:
            limitations.extend(assessment.limitations)

        items_by_claim: Dict[str, List[Any]] = defaultdict(list)
        for item in evidence_items:
            items_by_claim[item.claim].append(item)

        for claim in claims:
            claim_items = items_by_claim[claim.claim]
            unique_sources = {item.source_skill for item in claim_items}
            if len(unique_sources) == 1 and not any(
                claim.claim in limitation for limitation in limitations
            ):
                limitations.append(f"Claim relies on a single source: {claim.claim}")
            if all(item.evidence_kind == "model_prediction" for item in claim_items):
                limitations.append(f"Claim is supported only by predictive evidence: {claim.claim}")
        return limitations

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
