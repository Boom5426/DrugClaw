"""
Code Agent — generates and executes Python code to query drug knowledge skills.

The execution model is intentionally narrow:
  - query plan generation
  - constrained code generation
  - static validation
  - execution inside a proxy-only sandbox
  - automatic fallback to skill.retrieve() on any failure
"""
from __future__ import annotations

import ast
import io
import json
import signal
import traceback
from contextlib import redirect_stderr
from typing import Any, Dict, List, Optional

from .llm_client import LLMClient
from .skills.registry import SkillRegistry


ALLOWED_IMPORTS = {"json", "math", "re", "statistics"}
ALLOWED_BUILTINS = {
    "Exception", "ValueError", "TypeError", "KeyError", "IndexError",
    "len", "range", "min", "max", "sum", "sorted", "enumerate", "zip",
    "list", "dict", "set", "tuple", "str", "int", "float", "bool",
    "abs", "all", "any", "print",
}
BLOCKED_AST_NODES = (
    ast.While,
    ast.With,
    ast.AsyncWith,
    ast.AsyncFor,
    ast.AsyncFunctionDef,
    ast.ClassDef,
    ast.Global,
    ast.Nonlocal,
    ast.Lambda,
)
BLOCKED_NAMES = {
    "open", "eval", "exec", "compile", "__import__", "input", "help", "dir",
    "globals", "locals", "vars",
}
BLOCKED_MODULE_ROOTS = {
    "os", "sys", "subprocess", "shutil", "socket", "pathlib", "inspect",
    "importlib", "pickle", "glob", "tempfile", "builtins", "requests",
    "urllib", "http",
}
BLOCKED_ATTRIBUTE_NAMES = {
    "system", "popen", "fork", "spawn", "remove", "unlink", "rmdir", "environ",
    "putenv", "getenv", "listdir", "walk", "iterdir", "glob", "rglob", "mkdir",
    "makedirs", "write_text", "write_bytes", "read_text", "read_bytes", "chmod",
    "chown", "open", "write",
}
MAX_CODE_CHARS = 4000
MAX_AST_NODES = 200
MAX_OUTPUT_CHARS = 12000
MAX_RESULTS_PER_QUERY = 50
MAX_RECORD_VALUE_CHARS = 500


class OutputBudgetExceeded(RuntimeError):
    """Raised when generated code exceeds the output budget."""


class SafeSkillProxy:
    """Expose a narrow, query-oriented interface to generated code."""

    def __init__(
        self,
        skill: Any,
        *,
        entities: Dict[str, List[str]],
        query: str,
        max_results: int,
    ):
        self._skill = skill
        self._entities = {
            key: list(value) for key, value in (entities or {}).items()
        }
        self._query = query
        self._max_results = max_results
        self._last_records: List[Dict[str, Any]] = []

    def retrieve(self, max_results: int | None = None) -> List[Dict[str, Any]]:
        limit = min(max_results or self._max_results, self._max_results)
        records = self._skill.retrieve(
            entities=self._entities,
            query=self._query,
            max_results=limit,
        )
        self._last_records = [_sanitize_record(record) for record in records[:limit]]
        return list(self._last_records)

    @property
    def entities(self) -> Dict[str, List[str]]:
        return {
            key: list(value) for key, value in self._entities.items()
        }

    @property
    def query(self) -> str:
        return self._query

    @property
    def last_records(self) -> List[Dict[str, Any]]:
        return list(self._last_records)


class CoderAgent:
    """
    Agent that writes and executes Python code to query drug knowledge skills.

    Flow:
      1. Receive selected skill names + entity info from the Retriever Agent
      2. For each skill, read its description + example code
      3. Ask the LLM to write Python code that queries the skill for the
         specified entities
      4. Execute the code and capture stdout as the result string
      5. Aggregate results into a combined text block
    """

    def __init__(self, llm_client: LLMClient, skill_registry: SkillRegistry):
        self.llm = llm_client
        self.skill_registry = skill_registry
        self._last_execution_records: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Prompts
    # ------------------------------------------------------------------

    def get_system_prompt(self) -> str:
        return """You are the Code Agent of DrugClaw — a drug-specialized agentic RAG system operating inside a constrained query sandbox.

Your role is to write Python code that queries one drug knowledge resource for specific entities through a narrow safe interface.

You will be given:
1. A skill's description (name, subcategory, access mode, aim)
2. Example Python code showing how to use the skill's API
3. The entities to search for (drugs, genes, diseases, pathways)
4. The original user query for context
5. A query plan describing the intended transformations

You may use only these runtime objects:
- `skill.retrieve(max_results=...)`
- `safe_query(max_results=...)`
- `safe_filter_records(records, field, value)`
- `safe_sort_records(records, field, reverse=False)`
- `safe_format_output(records, fields=None, limit=10)`
- `entities`
- `query`

IMPORTANT:
- Use print() to output results — all printed text becomes the retrieval result
- Prefer `safe_query()` or `skill.retrieve()` over custom logic
- Keep code short and transformation-oriented
- Use only allowlisted imports: json, math, re, statistics
- Do NOT access the filesystem, environment variables, network, subprocesses, shells, or package installation
- Include source attribution (which database/resource the data came from)
- Handle cases where no results are found gracefully
- If the plan cannot be implemented safely, print a concise explanation instead of attempting risky code

Return ONLY the Python code, no markdown formatting, no explanation."""

    def get_query_plan_prompt(
        self,
        skill_name: str,
        entities: Dict[str, List[str]],
        query: str,
    ) -> str:
        entity_str = "\n".join(
            f"  {etype}: {', '.join(enames)}"
            for etype, enames in entities.items()
            if enames
        ) or "  (no specific entities extracted)"

        return f"""Create a compact JSON query plan for a constrained code sandbox.

Skill: {skill_name}
Entities:
{entity_str}
Original Query: {query}

Reply with JSON only:
{{
  "approach": "one-sentence plan",
  "operations": ["retrieve", "filter", "sort", "format"],
  "focus_fields": ["source_entity", "relationship", "target_entity", "evidence_text"],
  "output_style": "short human-readable summary",
  "needs_imports": []
}}"""

    def get_code_generation_prompt(
        self,
        skill_name: str,
        skill_info: str,
        query_plan_text: str,
        entities: Dict[str, List[str]],
        query: str,
    ) -> str:
        entity_str = "\n".join(
            f"  {etype}: {', '.join(enames)}"
            for etype, enames in entities.items()
            if enames
        )
        if not entity_str:
            entity_str = "  (no specific entities extracted)"

        return f"""Write Python code to query the following skill for the given entities.

=== Skill Info ===
{skill_info}

=== Query Plan ===
{query_plan_text}

=== Entities to Query ===
{entity_str}

=== Original Query ===
{query}

Write a short Python script that uses the provided sandbox helpers, queries this resource for the above entities, and prints the results.
Focus on the information most relevant to answering the original query. Prefer `safe_query()` plus filtering/sorting over anything more complex.
Return ONLY Python code."""

    # ------------------------------------------------------------------
    # Core execution
    # ------------------------------------------------------------------

    def generate_and_execute(
        self,
        skill_names: List[str],
        entities: Dict[str, List[str]],
        query: str,
        max_results_per_skill: int = 30,
    ) -> Dict[str, Any]:
        """
        Generate and execute query code for each skill.

        Returns a dict with:
          - "text": combined result text (str)
          - "per_skill": dict mapping skill_name -> {"plan": str, "code": str, "output": str, "error": str, "strategy": str}
        """
        all_outputs: List[str] = []
        per_skill: Dict[str, Dict[str, Any]] = {}

        for skill_name in skill_names:
            print(f"[Code Agent] Generating code for skill: {skill_name}")

            # Try the code-generation path first
            skill_info = self.skill_registry.get_skill_info_for_coder(skill_name)
            query_plan, code, output, error, records = self._generate_and_run_for_skill(
                skill_name, skill_info, entities, query, max_results_per_skill,
            )
            strategy = "constrained_code"

            if error:
                # Constrained code path failed — fallback to direct skill.retrieve()
                print(f"[Code Agent] Code execution failed for {skill_name}, "
                      f"falling back to skill.retrieve()")
                fallback_output, fallback_error, records = self._fallback_retrieve(
                    skill_name, entities, query, max_results_per_skill,
                )
                output = fallback_output
                error = fallback_error
                code = "(fallback: skill.retrieve())"
                strategy = "fallback_retrieve"

            per_skill[skill_name] = {
                "plan": query_plan,
                "code": code,
                "output": output,
                "error": error,
                "strategy": strategy,
                "records": records,
            }

            if output.strip():
                all_outputs.append(
                    f"=== Results from {skill_name} ===\n{output.strip()}\n"
                )
            else:
                all_outputs.append(
                    f"=== Results from {skill_name} ===\n"
                    f"(no results retrieved{'; error: ' + error if error else ''})\n"
                )

        combined_text = "\n".join(all_outputs)
        return {
            "text": combined_text,
            "per_skill": per_skill,
        }

    def _generate_and_run_for_skill(
        self,
        skill_name: str,
        skill_info: str,
        entities: Dict[str, List[str]],
        query: str,
        max_results_per_skill: int,
    ) -> tuple[str, str, str, str, List[Dict[str, Any]]]:
        """Generate a query plan, then constrained code, then execute it."""
        query_plan_text, plan_error = self._generate_query_plan(
            skill_name, entities, query,
        )
        if plan_error:
            return query_plan_text, "", "", plan_error, []

        messages = [
            {"role": "system", "content": self.get_system_prompt()},
            {"role": "user", "content": self.get_code_generation_prompt(
                skill_name, skill_info, query_plan_text, entities, query,
            )},
        ]
        try:
            code = self.llm.generate(messages, temperature=0.3)
        except Exception as exc:
            return query_plan_text, "", "", f"LLM error: {exc}", []

        # Clean up the code (remove markdown fences if any)
        code = self._clean_code(code)

        validation_error = self._validate_generated_code(code)
        if validation_error:
            return query_plan_text, code, "", validation_error, []

        output, error = self._execute_code(
            code,
            skill_name=skill_name,
            entities=entities,
            query=query,
            timeout_seconds=30,
            max_results=max_results_per_skill,
        )
        records = list(self._last_execution_records)
        if not error and not records:
            error = "generated code did not retrieve structured records"
        return query_plan_text, code, output, error, records

    def _generate_query_plan(
        self,
        skill_name: str,
        entities: Dict[str, List[str]],
        query: str,
    ) -> tuple[str, str]:
        messages = [
            {"role": "system", "content": "Produce JSON only."},
            {"role": "user", "content": self.get_query_plan_prompt(
                skill_name, entities, query,
            )},
        ]
        try:
            raw_plan = self.llm.generate(messages, temperature=0.2)
        except Exception as exc:
            return "", f"query plan error: {exc}"

        cleaned = self._clean_code(raw_plan)
        plan_error = self._validate_query_plan(cleaned)
        return cleaned, plan_error

    def _fallback_retrieve(
        self,
        skill_name: str,
        entities: Dict[str, List[str]],
        query: str,
        max_results: int,
    ) -> tuple[str, str, List[Dict[str, Any]]]:
        """Fallback: use the skill's retrieve() method directly."""
        skill = self.skill_registry.get_skill(skill_name)
        if skill is None:
            return "", f"Skill '{skill_name}' not registered", []

        try:
            results = skill.retrieve(
                entities=entities, query=query, max_results=max_results,
            )
            if not results:
                return "(no results)", "", []

            lines = []
            records = [_sanitize_record(result) for result in results[:max_results]]
            for r in results[:max_results]:
                line = (
                    f"[{r.source}] {r.source_entity} ({r.source_type}) "
                    f"--{r.relationship}--> "
                    f"{r.target_entity} ({r.target_type})"
                )
                if r.evidence_text:
                    line += f"\n  Evidence: {r.evidence_text}"
                if r.sources:
                    line += f"\n  Sources: {', '.join(r.sources[:3])}"
                lines.append(line)
            return "\n".join(lines), "", records
        except Exception as exc:
            return "", f"retrieve() error: {exc}", []

    # ------------------------------------------------------------------
    # Code execution
    # ------------------------------------------------------------------

    def _execute_code(
        self,
        code: str,
        *,
        skill_name: str,
        entities: Dict[str, List[str]],
        query: str,
        timeout_seconds: int = 30,
        max_results: int = 30,
    ) -> tuple[str, str]:
        """
        Execute generated Python code in a restricted namespace.

        Returns (stdout_output, error_message).
        """
        stderr_capture = io.StringIO()
        output_chunks: List[str] = []
        output_size = 0

        skill = self.skill_registry.get_skill(skill_name)
        if skill is None:
            return "", f"Skill '{skill_name}' not registered"

        safe_skill = SafeSkillProxy(
            skill,
            entities=entities,
            query=query,
            max_results=min(max_results, MAX_RESULTS_PER_QUERY),
        )
        self._last_execution_records = []

        def budget_print(*args, sep=" ", end="\n", **kwargs):
            nonlocal output_size
            if "file" in kwargs and kwargs["file"] is not None:
                raise ValueError("print(file=...) is not allowed in generated code")
            text = sep.join(str(arg) for arg in args) + end
            next_size = output_size + len(text)
            if next_size > MAX_OUTPUT_CHARS:
                raise OutputBudgetExceeded(
                    f"generated output exceeded {MAX_OUTPUT_CHARS} characters"
                )
            output_chunks.append(text)
            output_size = next_size

        exec_globals = {
            "__builtins__": self._safe_builtins(budget_print),
            "__name__": "__coder_agent__",
            "skill": safe_skill,
            "entities": safe_skill.entities,
            "query": safe_skill.query,
            "safe_query": safe_skill.retrieve,
            "safe_filter_records": safe_filter_records,
            "safe_sort_records": safe_sort_records,
            "safe_format_output": safe_format_output,
        }

        def _timeout_handler(signum, frame):
            raise TimeoutError(f"generated code exceeded {timeout_seconds}s")

        previous_handler = signal.getsignal(signal.SIGALRM) if hasattr(signal, "SIGALRM") else None

        try:
            previous_handler = signal.signal(signal.SIGALRM, _timeout_handler)
            signal.alarm(timeout_seconds)
            with redirect_stderr(stderr_capture):
                exec(code, exec_globals)
            signal.alarm(0)
            signal.signal(signal.SIGALRM, previous_handler)
            self._last_execution_records = safe_skill.last_records
            output = "".join(output_chunks)
            errors = stderr_capture.getvalue()
            return output, errors
        except Exception as exc:
            signal.alarm(0)
            if previous_handler is not None:
                signal.signal(signal.SIGALRM, previous_handler)
            self._last_execution_records = safe_skill.last_records
            output = "".join(output_chunks)
            error_msg = f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"
            return output, error_msg

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _clean_code(code: str) -> str:
        """Remove markdown code fences if present."""
        code = code.strip()
        if code.startswith("```python"):
            code = code[len("```python"):]
        elif code.startswith("```"):
            code = code[3:]
        if code.endswith("```"):
            code = code[:-3]
        return code.strip()

    @staticmethod
    def _validate_generated_code(code: str) -> str:
        """Static guardrail before executing generated code."""
        if len(code) > MAX_CODE_CHARS:
            return f"generated code exceeds {MAX_CODE_CHARS} characters"

        try:
            tree = ast.parse(code)
        except SyntaxError as exc:
            return f"generated code is not valid Python: {exc}"

        if sum(1 for _ in ast.walk(tree)) > MAX_AST_NODES:
            return f"generated code exceeds AST node budget ({MAX_AST_NODES})"

        saw_print = False

        for node in ast.walk(tree):
            if isinstance(node, BLOCKED_AST_NODES):
                return f"forbidden syntax in generated code: {type(node).__name__}"
            if isinstance(node, ast.Import):
                for alias in node.names:
                    root = alias.name.split(".")[0]
                    if root in BLOCKED_MODULE_ROOTS or root not in ALLOWED_IMPORTS:
                        return f"forbidden import in generated code: {root}"
            elif isinstance(node, ast.ImportFrom):
                module = (node.module or "").split(".")[0]
                if module in BLOCKED_MODULE_ROOTS or module not in ALLOWED_IMPORTS:
                    return f"forbidden import in generated code: {module}"
            elif isinstance(node, ast.Name):
                if node.id in BLOCKED_NAMES:
                    return f"forbidden call in generated code: {node.id}"
            elif isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    if node.func.id in BLOCKED_NAMES:
                        return f"forbidden call in generated code: {node.func.id}"
                    if node.func.id == "print":
                        saw_print = True
                elif isinstance(node.func, ast.Attribute):
                    if node.func.attr in BLOCKED_ATTRIBUTE_NAMES:
                        return f"forbidden attribute call in generated code: {node.func.attr}"
            elif isinstance(node, ast.Attribute):
                if node.attr in BLOCKED_ATTRIBUTE_NAMES:
                    return f"forbidden attribute access in generated code: {node.attr}"
                if isinstance(node.value, ast.Name) and node.value.id in BLOCKED_MODULE_ROOTS:
                    return f"forbidden attribute access in generated code: {node.value.id}.{node.attr}"

        if not saw_print:
            return "generated code must print retrieval output"
        return ""

    @staticmethod
    def _validate_query_plan(query_plan_text: str) -> str:
        try:
            plan = json.loads(query_plan_text)
        except json.JSONDecodeError as exc:
            return f"query plan is not valid JSON: {exc}"

        blocked_tokens = ("shell", "subprocess", "filesystem", "network", "os.", "open(")
        serialized = json.dumps(plan).lower()
        if any(token in serialized for token in blocked_tokens):
            return "query plan requests forbidden capabilities"

        return ""

    @staticmethod
    def _safe_builtins(print_fn) -> Dict[str, Any]:

        builtin_map = dict(__builtins__ if isinstance(__builtins__, dict) else __builtins__.__dict__)

        def _safe_import(name, globals=None, locals=None, fromlist=(), level=0):
            root = name.split(".")[0]
            if root not in ALLOWED_IMPORTS:
                raise ImportError(f"Import of '{name}' is not allowed in generated code")
            return __import__(name, globals, locals, fromlist, level)

        safe = {name: builtin_map[name] for name in ALLOWED_BUILTINS}
        safe["print"] = print_fn
        safe["__import__"] = _safe_import
        return safe


def _truncate_value(value: Any) -> Any:
    if isinstance(value, str):
        return value[:MAX_RECORD_VALUE_CHARS]
    if isinstance(value, list):
        return [_truncate_value(item) for item in value[:10]]
    if isinstance(value, dict):
        return {
            str(key): _truncate_value(item)
            for key, item in list(value.items())[:20]
        }
    return value


def _sanitize_record(record: Any) -> Dict[str, Any]:
    if hasattr(record, "to_dict"):
        raw = record.to_dict()
    elif isinstance(record, dict):
        raw = record
    else:
        raw = vars(record)

    safe_record = {
        str(key): _truncate_value(value)
        for key, value in raw.items()
    }
    return safe_record


def safe_filter_records(records: List[Dict[str, Any]], field: str, value: Any) -> List[Dict[str, Any]]:
    return [record for record in records if record.get(field) == value]


def safe_sort_records(
    records: List[Dict[str, Any]],
    field: str,
    reverse: bool = False,
) -> List[Dict[str, Any]]:
    return sorted(records, key=lambda record: str(record.get(field, "")), reverse=reverse)


def safe_format_output(
    records: List[Dict[str, Any]],
    fields: Optional[List[str]] = None,
    limit: int = 10,
) -> str:
    selected_fields = fields or [
        "source",
        "source_entity",
        "relationship",
        "target_entity",
        "evidence_text",
    ]
    lines: List[str] = []
    for record in records[:limit]:
        parts = []
        for field in selected_fields:
            value = record.get(field)
            if value in (None, "", [], {}):
                continue
            parts.append(f"{field}={value}")
        if parts:
            lines.append("; ".join(parts))
    return "\n".join(lines) if lines else "(no records)"
