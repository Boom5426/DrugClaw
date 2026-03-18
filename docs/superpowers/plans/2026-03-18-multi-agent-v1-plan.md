# Multi-Agent V1 Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a controlled Multi-Agent V1 upgrade with an explicit planner, structured claim assessment, and FSM-style orchestration while preserving the current CLI mode semantics and hardening guarantees.

**Architecture:** Introduce `QueryPlan` and `PlannerAgent` ahead of retrieval, add a structured `ClaimAssessment` layer between evidence collection and answer generation, and refactor the internal execution flow into explicit stages without changing external `simple` / `graph` / `web_only` behavior. Reuse the current resource registry, evidence models, Code Agent sandbox, and structured final answer path rather than replacing them.

**Tech Stack:** Python, dataclasses, LangGraph, pytest, existing DrugClaw agent modules and CLI.

---

## File Map

### New files

- `drugclaw/query_plan.py`
  Defines the `QueryPlan` schema and conservative fallback plan helpers.
- `drugclaw/agent_planner.py`
  Defines `PlannerAgent`, planner prompts, JSON normalization, and fallback behavior.
- `drugclaw/claim_assessment.py`
  Defines `ClaimAssessment`, verdict logic, and rule-based assessment helpers.
- `tests/planner/test_agent_planner.py`
  Covers planner classification, entity extraction normalization, and graph gating hints.
- `tests/assessment/test_claim_assessment.py`
  Covers claim verdicts, evidence ID preservation, and confidence behavior.
- `tests/system/test_orchestration_fsm.py`
  Covers stage transitions, graph gating, and graceful fallback behavior.

### Modified files

- `drugclaw/models.py`
  Adds `query_plan`, `claim_assessments`, and lightweight execution trace fields to shared state.
- `drugclaw/main_system.py`
  Adds planner and claim-assessment stages, explicit internal stage transitions, graph gating, and bounded fallback behavior.
- `drugclaw/agent_retriever.py`
  Removes duplicate planning responsibility and makes retrieval consume `QueryPlan`.
- `drugclaw/agent_responder.py`
  Refactors the responder to consume `ClaimAssessment[]` plus `EvidenceItem[]`.
- `drugclaw/agent_graph_builder.py`
  Accepts planner and state gating context, but remains a build-only component.
- `drugclaw/cli.py`
  Adds `--show-plan` and `--show-claims` while preserving current default output.
- `tests/cli/test_cli_output.py`
  Adds CLI debug output coverage for plans and claim assessments.
- `tests/responder/test_responder_final_answer.py`
  Updates responder expectations to flow through claim assessments.
- `tests/graph/test_graph_builder_grounding.py`
  Extends graph tests for gated usage and preserved grounding.

## Chunk 1: Query Planning Foundation

### Task 1: Add `QueryPlan` schema and fallback helpers

**Files:**
- Create: `drugclaw/query_plan.py`
- Modify: `drugclaw/models.py`
- Test: `tests/planner/test_agent_planner.py`

- [ ] **Step 1: Write the failing schema tests**

```python
from drugclaw.query_plan import QueryPlan, build_fallback_query_plan


def test_fallback_query_plan_is_conservative():
    plan = build_fallback_query_plan("What does imatinib target?")
    assert plan.question_type == "unknown"
    assert plan.requires_graph_reasoning is False
    assert plan.preferred_skills == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/planner/test_agent_planner.py::test_fallback_query_plan_is_conservative -v`
Expected: FAIL with import or attribute errors

- [ ] **Step 3: Implement `QueryPlan` and state fields**

```python
@dataclass
class QueryPlan:
    question_type: str
    entities: Dict[str, List[str]]
    subquestions: List[str]
    preferred_skills: List[str]
    preferred_evidence_types: List[str]
    requires_graph_reasoning: bool
    requires_prediction_sources: bool
    requires_web_fallback: bool
    answer_risk_level: str
    notes: List[str] = field(default_factory=list)
```

Add `query_plan`, `claim_assessments`, `execution_stage`, and `graph_decision_reason` to `AgentState`.

- [ ] **Step 4: Run targeted tests**

Run: `pytest tests/planner/test_agent_planner.py -q`
Expected: PASS for the new schema tests

- [ ] **Step 5: Commit**

```bash
git add drugclaw/query_plan.py drugclaw/models.py tests/planner/test_agent_planner.py
git commit -m "feat: add query plan schema"
```

### Task 2: Add `PlannerAgent`

**Files:**
- Create: `drugclaw/agent_planner.py`
- Modify: `drugclaw/models.py`
- Test: `tests/planner/test_agent_planner.py`

- [ ] **Step 1: Write failing planner tests**

```python
def test_planner_classifies_direct_target_lookup_without_graph():
    plan = PlannerAgent(llm_stub).plan("What does imatinib target?")
    assert plan.question_type == "target_lookup"
    assert plan.requires_graph_reasoning is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/planner/test_agent_planner.py::test_planner_classifies_direct_target_lookup_without_graph -v`
Expected: FAIL because `PlannerAgent` does not exist yet

- [ ] **Step 3: Implement minimal planner**

Implement:
- planner system prompt
- JSON normalization into `QueryPlan`
- conservative fallback when LLM output is invalid
- planner notes that remain concise and structured

- [ ] **Step 4: Expand planner tests**

Cover:
- target lookup
- labeling query
- DDI or mechanism query that should enable graph hints
- ambiguous query that still returns a valid plan

- [ ] **Step 5: Run planner test file**

Run: `pytest tests/planner/test_agent_planner.py -q`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add drugclaw/agent_planner.py tests/planner/test_agent_planner.py
git commit -m "feat: add planner agent"
```

## Chunk 2: Plan-Aware Retrieval and FSM Skeleton

### Task 3: Make retrieval consume `QueryPlan`

**Files:**
- Modify: `drugclaw/agent_retriever.py`
- Modify: `drugclaw/models.py`
- Test: `tests/planner/test_agent_planner.py`
- Test: `tests/system/test_orchestration_fsm.py`

- [ ] **Step 1: Write failing retrieval integration tests**

```python
def test_retriever_prefers_resource_filter_over_query_plan_hints():
    state = AgentState(
        original_query="What does imatinib target?",
        resource_filter=["BindingDB"],
        query_plan=QueryPlan(... preferred_skills=["ChEMBL", "DGIdb"])
    )
    updated = retriever.execute(state)
    assert "BindingDB" in updated.retrieved_text
```

- [ ] **Step 2: Run targeted failing tests**

Run: `pytest tests/system/test_orchestration_fsm.py::test_retriever_prefers_resource_filter_over_query_plan_hints -v`
Expected: FAIL because retriever still self-plans

- [ ] **Step 3: Refactor retriever**

Implement:
- `PlannerAgent` becomes source of primary plan
- retriever reads `state.query_plan`
- explicit `resource_filter` still wins
- plan `preferred_skills` is filtered through runtime registry availability
- entities are normalized from plan output instead of retriever-local JSON plan

- [ ] **Step 4: Run retrieval and regression tests**

Run: `pytest tests/planner/test_agent_planner.py tests/system/test_orchestration_fsm.py -q`
Expected: PASS for retrieval plan-consumption checks

- [ ] **Step 5: Commit**

```bash
git add drugclaw/agent_retriever.py tests/planner/test_agent_planner.py tests/system/test_orchestration_fsm.py
git commit -m "refactor: make retriever consume query plans"
```

### Task 4: Add FSM stages to `main_system.py`

**Files:**
- Modify: `drugclaw/main_system.py`
- Modify: `drugclaw/models.py`
- Test: `tests/system/test_orchestration_fsm.py`

- [ ] **Step 1: Write failing orchestration tests**

```python
def test_simple_mode_uses_claim_assessment_without_graph():
    result = system.query("What does imatinib target?", thinking_mode="simple")
    assert result["success"] is True
    assert result["execution_trace"] == ["PLAN", "RETRIEVE", "NORMALIZE_EVIDENCE", "ASSESS_CLAIMS", "ANSWER"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/system/test_orchestration_fsm.py::test_simple_mode_uses_claim_assessment_without_graph -v`
Expected: FAIL because execution stages are not explicit yet

- [ ] **Step 3: Implement FSM overlay**

Implement:
- explicit internal stage recording
- planner node before retrieval
- normalize-evidence step as explicit state transition
- graph node only in `graph` mode
- no open-ended loop in the main path

- [ ] **Step 4: Add bounded fallback behavior**

Implement:
- conservative planner fallback
- graph failure skips to claim assessment
- optional single fallback path only when allowed

- [ ] **Step 5: Run orchestration tests**

Run: `pytest tests/system/test_orchestration_fsm.py -q`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add drugclaw/main_system.py drugclaw/models.py tests/system/test_orchestration_fsm.py
git commit -m "refactor: add fsm-style orchestration"
```

## Chunk 3: Claim Assessment and Response Refactor

### Task 5: Add `ClaimAssessment` schema and scoring helpers

**Files:**
- Create: `drugclaw/claim_assessment.py`
- Test: `tests/assessment/test_claim_assessment.py`

- [ ] **Step 1: Write failing assessment tests**

```python
def test_claim_assessment_marks_conflicting_claim_as_uncertain():
    assessments = assess_claims(evidence_items)
    assert assessments[0].verdict == "uncertain"
    assert assessments[0].supporting_evidence_ids == ["E1"]
    assert assessments[0].contradicting_evidence_ids == ["E2"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/assessment/test_claim_assessment.py::test_claim_assessment_marks_conflicting_claim_as_uncertain -v`
Expected: FAIL because assessment module does not exist

- [ ] **Step 3: Implement the schema and rule-based judge**

Implement:
- `ClaimAssessment` dataclass
- grouping by claim text
- verdict mapping from evidence support directions
- confidence derived from current evidence scoring
- limitations populated for sparse, contradictory, or predictive-only evidence

- [ ] **Step 4: Run claim assessment tests**

Run: `pytest tests/assessment/test_claim_assessment.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add drugclaw/claim_assessment.py tests/assessment/test_claim_assessment.py
git commit -m "feat: add claim assessment layer"
```

### Task 6: Refactor responder to consume `ClaimAssessment[]`

**Files:**
- Modify: `drugclaw/agent_responder.py`
- Modify: `drugclaw/models.py`
- Test: `tests/responder/test_responder_final_answer.py`
- Test: `tests/assessment/test_claim_assessment.py`

- [ ] **Step 1: Write failing responder tests**

```python
def test_responder_uses_claim_assessments_in_final_answer():
    updated = responder.execute_simple(state_with_claim_assessments)
    assert updated.final_answer_structured.key_claims[0].claim == "Imatinib targets ABL1."
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/responder/test_responder_final_answer.py -q`
Expected: FAIL because responder still derives claims internally from raw evidence only

- [ ] **Step 3: Implement responder refactor**

Implement:
- `ResponderAgent` consumes `state.claim_assessments`
- fallback to internal assessment only for compatibility if assessments are absent
- `FinalAnswer` generation reuses assessed claim verdicts, evidence IDs, warnings, and limitations
- preserve existing `answer` and `final_answer_structured` return shapes

- [ ] **Step 4: Run responder and assessment tests**

Run: `pytest tests/responder/test_responder_final_answer.py tests/assessment/test_claim_assessment.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add drugclaw/agent_responder.py drugclaw/models.py tests/responder/test_responder_final_answer.py tests/assessment/test_claim_assessment.py
git commit -m "refactor: make responder consume claim assessments"
```

## Chunk 4: Graph Gating, CLI, and Final Verification

### Task 7: Gate graph reasoning and preserve grounding

**Files:**
- Modify: `drugclaw/main_system.py`
- Modify: `drugclaw/agent_graph_builder.py`
- Test: `tests/graph/test_graph_builder_grounding.py`
- Test: `tests/system/test_orchestration_fsm.py`

- [ ] **Step 1: Write failing graph-gating tests**

```python
def test_graph_mode_skips_graph_when_plan_and_evidence_do_not_require_it():
    result = system.query("What is the label information for metformin?", thinking_mode="graph")
    assert "OPTIONAL_GRAPH:skipped" in result["execution_trace"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/system/test_orchestration_fsm.py::test_graph_mode_skips_graph_when_plan_and_evidence_do_not_require_it -v`
Expected: FAIL because graph mode still enters graph path directly

- [ ] **Step 3: Implement graph gate**

Implement:
- planner recommendation gate
- evidence/entity-shape confirmation gate
- state trace that explains why graph ran or was skipped
- no changes to graph grounding semantics when graph does run

- [ ] **Step 4: Run graph and orchestration tests**

Run: `pytest tests/graph/test_graph_builder_grounding.py tests/system/test_orchestration_fsm.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add drugclaw/main_system.py drugclaw/agent_graph_builder.py tests/graph/test_graph_builder_grounding.py tests/system/test_orchestration_fsm.py
git commit -m "refactor: gate graph reasoning"
```

### Task 8: Add CLI debug visibility

**Files:**
- Modify: `drugclaw/cli.py`
- Test: `tests/cli/test_cli_output.py`

- [ ] **Step 1: Write failing CLI tests**

```python
def test_run_query_can_print_plan_and_claim_summaries(...):
    exit_code = cli._run_query(..., show_plan=True, show_claims=True)
    assert "question_type=" in captured.out
    assert "verdict=" in captured.out
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/cli/test_cli_output.py -q`
Expected: FAIL because CLI lacks new debug flags

- [ ] **Step 3: Implement minimal CLI additions**

Implement:
- `--show-plan`
- `--show-claims`
- summary printers that do not duplicate the main answer
- graph usage decision summary when available

- [ ] **Step 4: Run CLI tests**

Run: `pytest tests/cli/test_cli_output.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add drugclaw/cli.py tests/cli/test_cli_output.py
git commit -m "feat: add planner and claim debug cli output"
```

### Task 9: Final regression and README touch-up if needed

**Files:**
- Modify: `README.md` (only if CLI docs need updates)
- Modify: `README_CN.md` (only if CLI docs need updates)
- Test: full suite

- [ ] **Step 1: Update README only if new debug flags need documentation**

Add short usage notes for:
- `--show-plan`
- `--show-claims`

- [ ] **Step 2: Run focused and full verification**

Run:
- `pytest tests/planner/test_agent_planner.py tests/assessment/test_claim_assessment.py tests/system/test_orchestration_fsm.py -q`
- `pytest tests/cli/test_cli_output.py tests/responder/test_responder_final_answer.py tests/graph/test_graph_builder_grounding.py -q`
- `pytest -q`

Expected:
- new targeted tests pass
- existing hardening tests remain green

- [ ] **Step 3: Final commit**

```bash
git add README.md README_CN.md tests
git add drugclaw
git commit -m "feat: implement controlled multi-agent v1"
```

