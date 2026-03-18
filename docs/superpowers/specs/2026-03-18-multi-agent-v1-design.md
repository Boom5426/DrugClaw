# Multi-Agent V1 Design

**Date:** 2026-03-18

## Goal

Add a disciplined Multi-Agent V1 layer to DrugClaw that improves query understanding, routing, retrieval planning, and claim-level judgment without expanding product scope or weakening the current hardening guarantees.

This pass must preserve the external meaning of the existing CLI modes:

- `simple`
- `graph`
- `web_only`

The upgrade is internal and control-oriented. It is not a free-form autonomous multi-agent system.

## Current Baseline

The current post-hardening codebase already has:

- a centralized runtime resource registry
- structured evidence objects (`EvidenceItem`, `FinalAnswer`)
- inspectable confidence scoring
- a narrowed Code Agent execution surface
- evidence-aware responder behavior
- graph grounding support
- tests that cover the hardening changes

The current weak points are not raw capability gaps. They are orchestration gaps:

1. retrieval still starts from the raw query plus ad hoc retriever-side planning
2. claim judgment is still implicit inside responder logic
3. graph mode remains heavier than it needs to be for many queries
4. the high-level flow is split by mode, but not expressed as a clean finite-state machine

## Constraints

- Do not add new skills.
- Do not convert skills into agents.
- Do not add open-ended loops or free-form agent debate.
- Do not redesign the resource registry, Code Agent sandbox, or evidence model foundations.
- Preserve current CLI mode names, defaults, and broad user expectations.
- Prefer adapters and narrow refactors over broad rewrites.

## Options Considered

### Option 1: Planner + structured ClaimAssessment + FSM overlay

Introduce an explicit `QueryPlan`, a narrow `PlannerAgent`, a structured claim assessment step, and an FSM-style execution overlay while reusing the current retriever, responder, evidence, and graph components.

- Pros: best balance of control, traceability, and compatibility
- Cons: requires touching several core boundaries at once

### Option 2: Upgrade retriever planning only

Promote the retriever's current JSON planning into a first-class schema but keep the rest of the pipeline mostly unchanged.

- Pros: lowest implementation risk
- Cons: does not fully solve implicit claim judgment or orchestration clarity

### Option 3: Split planner, claim judge, and responder into separate LLM agents

Create a more explicitly agentic pipeline with multiple LLM-driven roles.

- Pros: most "multi-agent" on paper
- Cons: prompt surface grows significantly, control weakens, and testing becomes more fragile

## Selected Approach

Use Option 1.

This gives DrugClaw a pragmatic Multi-Agent V1 without discarding the hardening work. The system becomes more explicit about planning, graph use, and claim judgment, while the execution surface remains bounded and inspectable.

## High-Level Design

### New internal pipeline

For skill-based modes, the intended flow becomes:

`User Query -> PlannerAgent -> QueryPlan -> Retriever/Coder stack -> EvidenceItem[] -> ClaimAssessment step -> Responder -> FinalAnswer`

### FSM stages

The orchestration should be modeled with explicit stages:

- `PLAN`
- `RETRIEVE`
- `NORMALIZE_EVIDENCE`
- `OPTIONAL_GRAPH`
- `ASSESS_CLAIMS`
- `ANSWER`

These stages are internal control states, not new user-facing modes.

## Mode Semantics

The external mode contract remains unchanged.

### `simple`

Internal path:

`PLAN -> RETRIEVE -> NORMALIZE_EVIDENCE -> ASSESS_CLAIMS -> ANSWER`

No graph stage is entered.

### `graph`

Internal path:

`PLAN -> RETRIEVE -> NORMALIZE_EVIDENCE -> OPTIONAL_GRAPH -> ASSESS_CLAIMS -> ANSWER`

Graph work is allowed in this mode, but still gated. `graph` no longer means "always do expensive graph work no matter what." It means "graph reasoning is available and may be used when justified."

### `web_only`

Keep the existing direct web-search-oriented semantics. Do not force this mode through the skill retrieval planner path. Planning may still classify the query for traceability, but the mode must remain operationally distinct.

## QueryPlan

Add a dedicated schema, likely in `drugclaw/query_plan.py`.

### Required fields

- `question_type`
- `entities`
- `subquestions`
- `preferred_skills`
- `preferred_evidence_types`
- `requires_graph_reasoning`
- `requires_prediction_sources`
- `requires_web_fallback`
- `answer_risk_level`
- `notes`

### Semantics

`preferred_skills` is a hint, not an authority. It cannot override:

- explicit `resource_filter`
- runtime registry availability
- mode constraints

`requires_graph_reasoning` and `requires_web_fallback` are also recommendations from the planner, not absolute mandates.

## PlannerAgent

Add a narrow `PlannerAgent`, likely in `drugclaw/agent_planner.py`.

Responsibilities:

- classify the query
- extract entities
- decompose into subquestions
- recommend relevant skill and evidence families
- recommend whether graph reasoning is worth attempting
- flag whether prediction-oriented evidence may be needed

Non-responsibilities:

- no retrieval
- no answer synthesis
- no direct code execution
- no factual invention beyond planning assumptions

## Retriever Integration

The current retriever already performs lightweight planning. That duplication must be removed.

After this change:

- `PlannerAgent` owns primary planning
- `RetrieverAgent` consumes `QueryPlan`
- `RetrieverAgent` remains responsible for execution-oriented choices
- `CoderAgent` remains unchanged in role and safety posture

The retriever should adapt existing logic rather than being rewritten from scratch.

## ClaimAssessment

Add a dedicated schema, likely in `drugclaw/claim_assessment.py`.

### Required fields

- `claim`
- `verdict`
- `supporting_evidence_ids`
- `contradicting_evidence_ids`
- `confidence`
- `rationale`
- `limitations`

`verdict` must support:

- `supported`
- `contradicted`
- `uncertain`
- `insufficient`

## Claim Assessment Strategy

V1 should implement claim assessment as a structured step, not a separate standalone judge agent.

This is the lowest-risk design because:

- the responder already performs implicit claim aggregation
- evidence confidence rules already exist
- a structured assessment step can be tested without widening prompt roles too much

The responder should stop being the hidden judge. It should become the final presenter of already-assessed claims.

## Confidence Rules

Do not introduce a second unrelated scoring framework.

Claim-level confidence should be derived from the current evidence scoring rules and extended upward:

- multiple independent strong sources increase confidence
- contradictions reduce confidence
- sparse evidence reduces confidence
- prediction-only support remains lower confidence
- short or indirect evidence reduces confidence

The scoring path must remain inspectable in code and tests.

## Graph Gating

Graph work must become conditional.

### Gate requirements

Graph should run only when both conditions are satisfied:

1. the planner recommends graph reasoning
2. the evidence/entity shape suggests multi-hop or compositional reasoning is actually useful

### Typical positive triggers

- multi-entity relation questions
- mechanism or pathway composition
- DDI causal reasoning
- ADR causal or multi-step association reasoning

### Typical negative triggers

- direct labeling lookups
- single-record evidence retrieval
- straightforward ontology lookups
- simple direct target lookups that do not need relation composition

## Fallback Behavior

The old `reflect -> web_search -> retrieve` loop should be downgraded in V1.

Recommended V1 behavior:

- no open-ended reflection loop
- at most one explicit fallback path
- fallback only when mode and planner allow it
- fallback only when evidence is weak, sparse, or conflicting enough to justify it

If planning fails, use a conservative default plan.

If graph building fails, continue without graph.

If claim assessment is partial, still answer with limitations.

## State Model Changes

Add to `AgentState`:

- `query_plan`
- `claim_assessments`
- a lightweight graph decision trace such as `graph_decision_reason`

Avoid adding another large generic metadata bucket. The goal is more inspectable state, not less.

## CLI and Debug Visibility

Add optional debug visibility without changing default answer readability.

Suggested flags:

- `--show-plan`
- `--show-claims`

Existing `--show-evidence` should continue to work.

Default CLI output should remain concise. Debug views should expose:

- planner summary
- claim assessments
- graph gating decision

## File Plan

Likely new files:

- `drugclaw/query_plan.py`
- `drugclaw/agent_planner.py`
- `drugclaw/claim_assessment.py`

Likely modified files:

- `drugclaw/main_system.py`
- `drugclaw/models.py`
- `drugclaw/agent_retriever.py`
- `drugclaw/agent_responder.py`
- `drugclaw/agent_graph_builder.py`
- `drugclaw/cli.py`
- relevant tests

## Compatibility Risks

### Retriever drift

The current retriever both plans and executes. Extracting planning into `PlannerAgent` can accidentally change skill selection behavior if prompt responsibilities are not clearly split.

### Over-conservative answers

Making claim judgment explicit may improve honesty while making answers feel less assertive. This is acceptable only if the output remains readable and decision-useful.

### Graph-mode behavioral drift

`graph` mode semantics must remain intact externally even though graph execution becomes gated internally. This requires targeted regression coverage.

### Fallback tradeoff

Reducing the old reflection loop to a bounded fallback may miss some marginal evidence gains, but this is an intentional tradeoff in favor of control and predictability.

## Verification Plan

At minimum, verify:

1. planner tests for stable classification and non-overactive graph triggers
2. retriever tests proving `QueryPlan` is consumed
3. claim assessment tests for supported, contradicted, uncertain, and insufficient verdicts
4. orchestration tests for `simple`, `graph`, and `web_only` transitions
5. CLI tests for `--show-plan`, `--show-claims`, and unchanged default output
6. regression tests proving evidence IDs and structured outputs survive end to end
7. existing hardening tests still pass

## Out of Scope

- free-form autonomous planning loops
- agent debate
- skill-per-agent conversion
- complex memory systems
- broader Code Agent redesign
- resource registry redesign
- changes to the external CLI mode contract

## Expected Outcome

After this pass:

- the system will expose an explicit `QueryPlan`
- retrieval will be plan-aware rather than query-only
- claim judgment will be explicit and inspectable
- graph usage will be conditional and justified
- orchestration will be more controlled without becoming more complicated for users

