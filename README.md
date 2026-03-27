# DrugClaw

<p align="center">
  <img src="./support/DrugClaw_Logo.png" alt="DrugClaw Logo" width="520" />
</p>

<p align="center">
  <strong>Agentic RAG for Drug Knowledge Retrieval, Reasoning, and Evidence Synthesis</strong>
</p>

<p align="center">  
  <a href="./DrugClaw_Paper.pdf">项目技术报告</a>  
  <a href="./README_CN.md">中文文档 / Chinese Version</a>
</p>

<p align="center">
  <img alt="Domain" src="https://img.shields.io/badge/Domain-Drug%20Intelligence-1f6feb">
  <img alt="Registry" src="https://img.shields.io/badge/Registry-Source%20of%20Truth-0a7f5a">
  <img alt="Skills" src="https://img.shields.io/badge/Skills-Registry%20Driven-f59e0b">
  <img alt="Modes" src="https://img.shields.io/badge/Modes-GRAPH%20%7C%20SIMPLE%20%7C%20WEB__ONLY-7c3aed">
</p>

DrugClaw is a drug-centered multi-agent RAG system designed for queries that generic assistants often handle poorly: drug targets, adverse drug reactions, drug-drug interactions, mechanisms of action, pharmacogenomics, repurposing, and evidence synthesis across heterogeneous biomedical resources.

It is not a generic RAG stack with a biomedical prompt on top. DrugClaw is opinionated around drug-native tasks from resource organization to retrieval strategy, reasoning flow, and final answer structure.

## Why DrugClaw

Most biomedical QA systems stop at "retrieve a few passages and summarize them." Drug questions are harder: they require precise handling of target evidence, ADR provenance, DDI mechanisms, labeling details, and PGx constraints. Some tools connect many databases but flatten them into a single rigid interface; others optimize for conversational UX while relying on weak, thin, or poorly traceable evidence.

- Organizes drug resources through a **registry-backed 15-subcategory skill tree**
- Uses a **Code Agent** to query each source in its native style instead of forcing one rigid abstraction
- Supports **graph-based reasoning** for multi-hop evidence synthesis
- Keeps **Web Search** as a fallback for recent literature and external evidence
- Built around **drug-native tasks**, not generic biomedical branding

The runtime resource registry is the source of truth for what is currently enabled, degraded, missing local metadata, or disabled. Availability depends on the environment, local files under `resources_metadata/`, optional dependencies, and API reachability.

In short, DrugClaw is not trying to be just another fluent assistant. Its goal is to raise resource density, retrieval fidelity, and evidence-grounded reasoning at the same time.

## Quick Start

Run the commands below from the cloned repository root.

### 1. Create a `.venv` and install dependencies

```bash
python3 -m venv .venv
. .venv/bin/activate  # Windows: `.venv\\Scripts\\activate`
python -m pip install --upgrade pip
python -m pip install --no-build-isolation -r requirements.txt
```

`requirements.txt` installs the core package, test dependencies, common local-example libraries, and the CLI-first skill extras that were previously listed separately.

If you prefer the smaller editable install only:

```bash
python -m pip install -e .[dev] --no-build-isolation
```

### 2. Prepare `navigator_api_keys.json`

DrugClaw uses any **OpenAI-compatible** API endpoint. This includes OpenAI, Azure OpenAI, LLaMA served via vLLM or Ollama, and other OpenAI-compatible providers.

First copy the template:

```bash
cp navigator_api_keys.example.json navigator_api_keys.json
```

Then fill in your real credentials:

```json
{
  "api_key": "your-api-key-here",
  "base_url": "https://your-endpoint.com/v1",
  "model": "gpt-4o",
  "max_tokens": 20000,
  "timeout": 60,
  "temperature": 0.7
}
```

**Example configurations for common providers:**

| Provider | `base_url` | `model` |
| --- | --- | --- |
| OpenAI | `https://api.openai.com/v1` | `gpt-4o`, `gpt-4o-mini` |
| Azure OpenAI | `https://YOUR.openai.azure.com/v1` | your deployment name |
| vLLM (local LLaMA) | `http://localhost:8000/v1` | `meta-llama/Llama-3.1-8B-Instruct` |
| Ollama | `http://localhost:11434/v1` | `llama3.1`, `qwen2.5` |
| Together AI | `https://api.together.xyz/v1` | `meta-llama/Llama-3.1-70B-Instruct-Turbo` |

If you must keep a non-default filename such as `api_keys.json`, pass it explicitly with `--key-file api_keys.json`.

### 3. Run the official CLI demo

This is the recommended first experience. At this stage you only need a working LLM config; local resource packs can wait until later.

You can run it without installation:

```bash
python -m drugclaw list
python -m drugclaw doctor
python -m drugclaw demo
```

The default demo uses:

- `SIMPLE` mode
- online labeling-style resources
- a metformin labeling and safety query

You can also run your own query:

```bash
python -m drugclaw run --query "What are the known drug targets of imatinib?"
```

DrugClaw also includes a minimal web/API service entrypoint:

```bash
python -m drugclaw serve --host 127.0.0.1 --port 8000
```

After startup, open:

```text
http://127.0.0.1:8000/
```

The first service version is intentionally conservative:

- it wraps the existing `DrugClawSystem.query()` flow instead of replacing it
- it defaults to low concurrency to avoid destabilizing the main workload
- it does not implement custom token quotas yet and relies on upstream account limits

If you want to inspect the internal planner, claim summaries, or agent logs while debugging:

```bash
python -m drugclaw run --query "What does imatinib target?" --show-plan --show-claims
python -m drugclaw run --query "What prescribing and safety information is available for metformin?" --debug-agents
```

If you want to save a local Markdown report for your own custom query:

```bash
python -m drugclaw run --query "What does imatinib target?" --save-md-report
```

This writes a Markdown report to `query_logs/<query_id>/report.md` and prints the saved path after the run finishes.

Phase 1 now includes input normalization for canonical drug names and common aliases. That means inputs such as `imatinib` / `Gleevec` or `metformin` / `Glucophage` are normalized to the same canonical drug before planning and retrieval.

Phase 2 also adds structured identifier normalization at query entry for:

- `ChEMBL ID`
- `PubChem CID`
- `InChIKey`

These identifiers are resolved to a canonical drug name before planning and retrieval. Resolution details are persisted to `query_logs/<query_id>/metadata.json` under `input_resolution.identifier_resolution`.

You can verify alias equivalence locally with these pairs:

```bash
python -m drugclaw run --query "What are the known drug targets of imatinib?" --mode simple --resource-filter "ChEMBL,DGIdb,Open Targets Platform" --show-plan --show-claims --save-md-report
python -m drugclaw run --query "What are the known drug targets of Gleevec?" --mode simple --resource-filter "ChEMBL,DGIdb,Open Targets Platform" --show-plan --show-claims --save-md-report

python -m drugclaw run --query "What prescribing and safety information is available for metformin?" --mode simple --resource-filter "DailyMed,openFDA Human Drug,MedlinePlus Drug Info" --show-plan --show-claims --save-md-report
python -m drugclaw run --query "What prescribing and safety information is available for Glucophage?" --mode simple --resource-filter "DailyMed,openFDA Human Drug,MedlinePlus Drug Info" --show-plan --show-claims --save-md-report
```

When comparing the two runs, focus on:

- whether `show-plan` resolves both inputs to the same canonical drug entity
- whether `query_logs/<query_id>/metadata.json` shows the same `normalized_query` and `input_resolution.canonical_drug_names`
- whether `report.md` and the final answer keep the same core conclusions and sources

The built-in phase-1 alias seed currently includes:

- `imatinib` ↔ `Gleevec`
- `metformin` ↔ `Glucophage`
- `sildenafil` ↔ `Viagra`
- `atorvastatin` ↔ `Lipitor`

You can also verify structured identifier equivalence with the same drug:

```bash
python -m drugclaw run --query "What are the known drug targets of CHEMBL941?" --mode simple --resource-filter "ChEMBL,DGIdb,Open Targets Platform" --show-plan --show-claims --save-md-report
python -m drugclaw run --query "What are the known drug targets of PubChem CID 5291?" --mode simple --resource-filter "ChEMBL,DGIdb,Open Targets Platform" --show-plan --show-claims --save-md-report
python -m drugclaw run --query "What are the known drug targets of KTUFNOKKBVMGRW-UHFFFAOYSA-N?" --mode simple --resource-filter "ChEMBL,DGIdb,Open Targets Platform" --show-plan --show-claims --save-md-report
```

For these runs, check:

- whether `show-plan` converges on the same canonical drug entity
- whether `metadata.json` contains the expected `input_resolution.identifier_resolution` trace
- whether `report.md` keeps the same core target list and evidence sources as the name-based query

If the demo runs successfully, you already have a minimal usable setup. The next step is optional and only matters when you want broader coverage from `LOCAL_FILE` skills and local datasets.

### 3.1 Minimal Web/API service

The web service exposes a small set of endpoints for internal or small-scale usage:

- `GET /health`
- `GET /resources`
- `POST /api/query`
- `GET /api/queries/{query_id}`
- `GET /api/queries/{query_id}/report`

Quick local validation:

```bash
curl http://127.0.0.1:8000/health
curl -X POST http://127.0.0.1:8000/api/query \
  -H 'Content-Type: application/json' \
  -d '{"query":"What are the known drug targets of imatinib?","mode":"simple","resource_filter":["ChEMBL","DGIdb","Open Targets Platform"],"save_md_report":true}'
```

Deployment recommendation for the first serviceized version:

- run the app with `drugclaw serve`
- place `nginx` in front of it
- use `systemd` or another process supervisor
- keep the service-side API key private; never expose it to the browser

If you want to expose this service publicly, start with:

- `docs/2026-03-23-DrugClaw-公网部署说明.md`
- `deploy/nginx/drugclaw.conf`
- `deploy/systemd/drugclaw.service`

Do not expose port `8000` directly to the public Internet for production use. Keep DrugClaw bound to `127.0.0.1:8000` and publish it through `nginx + HTTPS`.

### 4. Prepare local resources under `resources_metadata/` for broader coverage

Many skills use `LOCAL_FILE` access mode. Those resources are not required for the first demo, but they improve coverage and unlock skills that depend on local datasets.

Recommended resolution order:

- Use files already present under `resources_metadata/...`
- If missing, sync from the maintained mirror first
- Only fall back to the original source site if the mirror does not contain the data
- Do not commit private credentials, local snapshots, or temporary downloads under `resources_metadata/`; only keep curated minimal fixtures that are required for tests

Maintained mirror:

- `https://huggingface.co/datasets/Mike2481/DrugClaw_resources_data`

Directory examples:

- `resources_metadata/dti/...`
- `resources_metadata/adr/...`
- `resources_metadata/drug_knowledgebase/...`
- `resources_metadata/drug_repurposing/...`
- `resources_metadata/ddi/...`

If some old `SKILL.md`, `example.py`, or archived docs still show absolute paths, treat them as legacy examples. The active convention is the repository-local `resources_metadata/...` layout.

Downloading local resources is recommended if you want more stable retrieval from dataset-backed skills and better overall coverage than the minimal online-first demo path.

### 5. Use the installed `drugclaw` command

```bash
git config core.hooksPath .githooks
drugclaw list
drugclaw doctor
drugclaw demo
drugclaw run --query "What are the known drug targets of imatinib?"
```

### 6. Optional examples and compatibility entrypoints

The main entrypoint is still the CLI. If you want sample scripts, they now live under `examples/`.

The lightweight demo wrapper is:

```bash
python examples/run_minimal.py
```

You can also pass CLI arguments through it:

```bash
python examples/run_minimal.py demo --preset label
python examples/run_minimal.py run --query "What prescribing and safety information is available for metformin?"
```

### 7. Run environment checks when you need diagnosis

```bash
python -m drugclaw doctor
```

It checks:

- whether `navigator_api_keys.json` exists and is complete
- whether `langgraph` and `openai` are importable
- whether built-in demo presets have the resources they need
- whether the `drugclaw` command is installed
- whether repository Git hooks are enabled

### 8. Enable secret-protection Git hooks

```bash
git config core.hooksPath .githooks
```

These hooks block committing API key files.

### 9. List built-in demos and recommended entrypoints

```bash
python -m drugclaw list
```

It shows:

- built-in demo presets
- supported thinking modes
- recommended first commands
- common resource filter combinations

### 10. Call DrugClaw from Python

```python
from drugclaw.config import Config
from drugclaw.main_system import DrugClawSystem
from drugclaw.models import ThinkingMode

config = Config(key_file="navigator_api_keys.json")
system = DrugClawSystem(config)

result = system.query(
    "What prescribing and safety information is available for metformin?",
    thinking_mode=ThinkingMode.SIMPLE,
    resource_filter=["DailyMed", "openFDA Human Drug", "MedlinePlus Drug Info"],
)

print(result["answer"])
print(result["normalized_query"])
print(result["input_resolution"])
```

### 11. Thinking modes

```python
from drugclaw.models import ThinkingMode

system.query("...", thinking_mode=ThinkingMode.GRAPH)
system.query("...", thinking_mode=ThinkingMode.SIMPLE)
system.query("...", thinking_mode=ThinkingMode.WEB_ONLY)
```

## Highlights

<p align="center">
  <img src="./support/DrugClaw.png" alt="DrugClaw Overview" width="760" />
</p>

### 1. Vibe-coding retrieval

Each skill ships with its own `SKILL.md` and `example.py`. The Code Agent reads both, learns the source's native usage pattern, and generates query code dynamically for the current task.

That means DrugClaw does not require every database, API, or dataset to pretend to be the same thing.

For `LOCAL_FILE` skills, the recommended default behavior is:

- check `resources_metadata/...` first
- if missing, guide the user to the maintained Hugging Face mirror
- do not assume the original download endpoint is still reliable

### 2. Organized around drug tasks

DrugClaw covers all 15 subcategories through the runtime resource registry:

- drug targets and activity (DTI)
- adverse drug reactions and pharmacovigilance (ADR)
- drug knowledgebases
- mechanisms of action
- labeling and prescribing information
- ontology and normalization
- drug repurposing
- pharmacogenomics
- drug-drug interactions
- drug toxicity
- drug combinations
- drug molecular properties
- drug-disease associations
- patient reviews
- drug NLP / text mining

### 3. Three working modes

- `GRAPH`: retrieve -> graph build -> rerank -> respond -> reflect
- `SIMPLE`: retrieve and answer directly
- `WEB_ONLY`: use only online search and literature retrieval

### 4. Built for evidence synthesis

DrugClaw is suitable for questions such as:

- "What are the known targets, adverse effects, and interaction risks of imatinib?"
- "Which approved drugs may be repurposed for triple-negative breast cancer?"
- "What pharmacogenomic guidance exists for clopidogrel and CYP2C19?"
- "Are there clinically meaningful interactions between warfarin and NSAIDs?"

## Architecture

```text
User Query
   |
   v
Retriever Agent
   |- navigates the 15-subcategory skill tree
   |- extracts key entities
   |- selects relevant resources
   |
   v
Code Agent
   |- reads SKILL.md + example.py
   |- writes custom query code  (vibe coding)
   |- executes resource-specific retrieval
   |
   |- [on code failure] fallback to fixed retrieve.py CLI
   |     retrieve.py entity1 [entity2 ...]
   |     (deterministic, no LLM, uses example.py directly)
   |
   +--> SIMPLE mode --> Responder --> Final Answer
   |
   +--> GRAPH mode
         -> Graph Builder
         -> Reranker
         -> Responder
         -> Reflector
         -> optional Web Search
         -> Final Answer
```

## Skill Fallback Mechanism

Each skill directory that has an `example.py` also includes a `retrieve.py` —
a fixed, deterministic CLI script for direct entity retrieval without LLM code
generation.

**When is `retrieve.py` used?**

The Code Agent first attempts *vibe coding*: it reads `SKILL.md` and
`example.py`, then generates and executes a custom Python script to query the
skill. If that generated code fails (validation error, runtime exception,
timeout, or no structured records returned), the Code Agent automatically falls
back to running `retrieve.py` via subprocess.

**How to run `retrieve.py` manually:**

```bash
# Query a single skill by entity name
python skills/dti/chembl/retrieve.py imatinib
python skills/adr/faers/retrieve.py aspirin metformin
python skills/ddi/ddinter/retrieve.py warfarin aspirin

# Any number of entity arguments is accepted
python skills/drug_toxicity/livertox/retrieve.py acetaminophen isoniazid
```

Input: one or more entity names as positional command-line arguments.
Output: human-readable text printed to stdout, suitable as LLM context.

**Fallback priority order:**
1. Vibe coding (LLM-generated code via Code Agent)
2. `retrieve.py` subprocess (fixed deterministic script)
3. `skill.retrieve()` method (last resort, always available)

## Registry Inspection

Use the CLI to inspect the current registry summary and per-resource status:

```bash
python -m drugclaw list
python -m drugclaw doctor
```

`list` shows registry-derived totals and a status line for each resource. `doctor` explains why a resource is unavailable, including missing metadata paths and missing dependencies when detectable.

If you want the human-readable answer plus the structured claim/evidence summary, use:

```bash
python -m drugclaw run --query "What does imatinib target?" --show-evidence
```

## Implemented Skills

| Category | Skills |
| --- | --- |
| DTI | ChEMBL, BindingDB, DGIdb, Open Targets Platform, TTD, STITCH, TarKG, GDKD, Molecular Targets, Molecular Targets Data |
| ADR | FAERS, SIDER, nSIDES, ADReCS |
| Drug Knowledgebase | UniD3, DrugBank, IUPHAR/BPS, DrugCentral, CPIC, PharmKG, WHO Essential Medicines List, FDA Orange Book |
| Drug Mechanism | DRUGMECHDB |
| Drug Labeling | openFDA Human Drug, DailyMed, MedlinePlus Drug Info |
| Drug Ontology | RxNorm, ChEBI, ATC/DDD, NDF-RT |
| Drug Repurposing | RepoDB, DRKG, OREGANO, Drug Repurposing Hub, DrugRepoBank, RepurposeDrugs |
| Pharmacogenomics | PharmGKB |
| DDI | MecDDI, DDInter, KEGG Drug |
| Drug Toxicity | UniTox, LiverTox, DILIrank, DILI |
| Drug Combination | DrugCombDB, DrugComb |
| Drug Molecular Property | GDSC |
| Drug Disease | SemaTyP |
| Drug Review | WebMD Drug Reviews, Drug Reviews (Drugs.com) |
| Drug NLP | DDI Corpus 2013, DrugProt, ADE Corpus, CADEC, PsyTAR, TAC 2017 ADR, PHEE |

`WebSearch` is also available as an external retrieval supplement built around DuckDuckGo + PubMed style search.

## Repository Layout

```text
README.md / README_CN.md
  GitHub-facing entry docs

examples/
  Optional runnable examples and compatibility wrappers

scripts/legacy/
  Historical maintainer helpers, not part of the public interface

drugclaw/
  Package runtime and CLI entrypoints

skills/
  <subcategory>/<skill_name>/
    *_skill.py
    example.py
    SKILL.md
    README.md

skillexamples/
  Resource-specific usage examples and operator notes

tools/
  Maintainer smoke checks and resource validation scripts

resources_metadata/
  local data files
```

For a short contributor-facing directory guide, see `docs/repository-guide.md`.

## Differentiation

### Resource-native querying instead of forced abstraction

DrugClaw does not require every biomedical source to be flattened into a single interface.

### Graph reasoning instead of flat summarization

DrugClaw can turn free-form retrieval results into triples, subgraphs, ranked paths, and evidence-aware answers rather than simply stitching together excerpts.

### Drug-first scope instead of generic biomedical positioning

This system is built around drug tasks: DTI, ADR, DDI, labeling, repurposing, PGx, and mechanism reasoning.

## Current Notes

- The repository can be imported directly from the project root.
- `pyproject.toml` is now aligned with the current package layout for local CLI usage.
- Some skills still depend on local files under `resources_metadata/`.
- Multi-iteration `GRAPH` behavior still depends on further configuration such as `MAX_ITERATIONS`.

## Citation

If you use DrugClaw in research or product work, cite this repository and the upstream resources used by the selected skills.


## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=QSong-github/DrugClaw&type=Date)](https://star-history.com/#QSong-github/DrugClaw&Date)


## License

MIT License
