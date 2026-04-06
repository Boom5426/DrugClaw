"""
Query Logger - Folder-based per-query storage with Markdown reports.

Each query is stored in its own directory:

    query_logs/
    ├── query_index.json                        # Global index
    └── query_20260317_120000_123456/           # One folder per query
        ├── answer.md                           # Rich Markdown answer card
        ├── report.md                           # Optional saved Markdown export
        ├── metadata.json                       # Query metadata + metrics
        ├── reasoning_trace.md                  # Step-by-step reasoning
        ├── evidence.json                       # Structured evidence records
        └── full_result.pkl                     # Complete pickle dump
"""
from __future__ import annotations

import contextlib
import json
import os
import pickle
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import fcntl
except ImportError:  # pragma: no cover - only relevant on non-POSIX platforms
    fcntl = None

from .response_formatter import (
    format_reasoning_trace,
    format_source_citations,
    wrap_answer_card,
)


class QueryLogger:
    """
    Manages persistent, folder-based storage of query history.
    Each query gets its own directory with structured Markdown + JSON files.
    """

    def __init__(self, log_dir: str = "./query_logs"):
        self.log_dir = Path(log_dir).resolve()
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Global index for fast lookup
        self.index_file = self.log_dir / "query_index.json"
        self._load_or_create_index()

    # ------------------------------------------------------------------
    # Index management
    # ------------------------------------------------------------------

    def _load_or_create_index(self):
        if self.index_file.exists():
            self.index = self._read_index_from_disk()
        else:
            self.index = {"total_queries": 0, "queries": []}
            self._save_index()
        self._sync_index_from_logs()

    def _save_index(self):
        self._write_index_to_disk(self.index)

    def _read_index_from_disk(self) -> Dict[str, Any]:
        if not self.index_file.exists():
            return {"total_queries": 0, "queries": []}
        with open(self.index_file, "r", encoding="utf-8") as f:
            payload = json.load(f)
        return self._normalize_index(payload)

    @staticmethod
    def _normalize_index(payload: Dict[str, Any] | None) -> Dict[str, Any]:
        raw_queries = list((payload or {}).get("queries", []) or [])
        normalized_queries: List[Dict[str, Any]] = []
        seen: set[str] = set()
        for entry in raw_queries:
            query_id = str((entry or {}).get("query_id", "") or "").strip()
            if not query_id or query_id in seen:
                continue
            seen.add(query_id)
            normalized_queries.append(dict(entry or {}))
        return {
            "total_queries": len(normalized_queries),
            "queries": normalized_queries,
        }

    @contextlib.contextmanager
    def _locked_index_handle(self, mode: str):
        with open(self.index_file, mode, encoding="utf-8") as f:
            if fcntl is not None:
                fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            try:
                yield f
            finally:
                f.flush()
                os.fsync(f.fileno())
                if fcntl is not None:
                    fcntl.flock(f.fileno(), fcntl.LOCK_UN)

    def _write_index_to_disk(self, payload: Dict[str, Any]) -> None:
        normalized = self._normalize_index(payload)
        with self._locked_index_handle("a+") as f:
            f.seek(0)
            f.truncate()
            json.dump(normalized, indent=2, fp=f, ensure_ascii=False)
        self.index = normalized

    def _append_index_entry(self, entry: Dict[str, Any]) -> None:
        self.index_file.parent.mkdir(parents=True, exist_ok=True)
        if not self.index_file.exists():
            self.index_file.write_text(
                json.dumps({"total_queries": 0, "queries": []}, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        with self._locked_index_handle("r+") as f:
            try:
                current = json.load(f)
            except json.JSONDecodeError:
                current = {"total_queries": 0, "queries": []}
            normalized = self._normalize_index(current)
            known_ids = {
                str(item.get("query_id", "") or "").strip()
                for item in normalized.get("queries", [])
            }
            query_id = str(entry.get("query_id", "") or "").strip()
            if query_id and query_id not in known_ids:
                normalized["queries"].append(dict(entry))
            normalized["total_queries"] = len(normalized["queries"])
            f.seek(0)
            f.truncate()
            json.dump(normalized, indent=2, fp=f, ensure_ascii=False)
        self.index = normalized

    def _sync_index_from_logs(self) -> None:
        entries_by_id = {
            str(entry.get("query_id", "") or "").strip(): dict(entry)
            for entry in self.index.get("queries", [])
            if str(entry.get("query_id", "") or "").strip()
        }
        changed = False
        for meta_file in sorted(self.log_dir.glob("query_*/metadata.json")):
            meta = json.loads(meta_file.read_text(encoding="utf-8"))
            query_id = str(meta.get("query_id", "") or "").strip()
            if not query_id or query_id in entries_by_id:
                continue
            entries_by_id[query_id] = {
                "query_id": query_id,
                "timestamp": str(meta.get("timestamp", "") or ""),
                "query": str(meta.get("query", "") or "")[:100],
                "success": bool(meta.get("success", False)),
                "iterations": int(meta.get("iterations", 0) or 0),
                "mode": str(meta.get("mode", "") or ""),
            }
            changed = True
        if not changed:
            self.index = self._normalize_index(self.index)
            return
        merged_queries = sorted(
            entries_by_id.values(),
            key=lambda entry: str(entry.get("timestamp", "") or ""),
        )
        self._write_index_to_disk(
            {
                "total_queries": len(merged_queries),
                "queries": merged_queries,
            }
        )

    # ------------------------------------------------------------------
    # Core logging
    # ------------------------------------------------------------------

    def log_query(
        self,
        query: str,
        result: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None,
        save_md_report: bool = False,
    ) -> str:
        """
        Log a query and its results into a dedicated folder.

        Returns the query_id (also the folder name).
        """
        timestamp = datetime.now()
        query_id = f"query_{timestamp.strftime('%Y%m%d_%H%M%S_%f')}"

        # Create query folder
        query_dir = self.log_dir / query_id
        query_dir.mkdir(parents=True, exist_ok=True)

        answer = result.get("answer", "")

        # ── 1. answer.md — rich Markdown answer card ────────────────
        md_answer = wrap_answer_card(answer, result, timestamp)
        (query_dir / "answer.md").write_text(md_answer, encoding="utf-8")

        if save_md_report:
            (query_dir / "report.md").write_text(md_answer, encoding="utf-8")

        # ── 2. metadata.json — query metadata + metrics ─────────────
        meta = {
            "query_id": query_id,
            "timestamp": timestamp.isoformat(),
            "query": query,
            "normalized_query": result.get("normalized_query", query),
            "resolved_entities": result.get("resolved_entities", {}),
            "input_resolution": result.get("input_resolution", {}),
            "mode": result.get("mode", ""),
            "resource_filter": result.get("resource_filter", []),
            "iterations": result.get("iterations", 0),
            "evidence_graph_size": result.get("evidence_graph_size", 0),
            "final_reward": result.get("final_reward", 0.0),
            "success": result.get("success", False),
            "reasoning_summary": [
                {
                    "step": s.get("step", 0),
                    "reward": s.get("reward", 0.0),
                    "evidence_sufficiency": s.get("evidence_sufficiency", 0.0),
                }
                for s in result.get("reasoning_history", [])
            ],
            "user_metadata": metadata or {},
        }
        runtime_metadata = self._extract_runtime_metadata(metadata or {})
        if runtime_metadata:
            meta["runtime_metadata"] = runtime_metadata
            if runtime_metadata.get("suite_name"):
                meta["suite_name"] = runtime_metadata["suite_name"]
        latest_scorecard = self.get_latest_scorecard_summary()
        if latest_scorecard:
            meta["recent_scorecard"] = latest_scorecard
        query_plan_summary = self._summarize_query_plan(result.get("query_plan"))
        if query_plan_summary:
            meta["query_plan_summary"] = query_plan_summary
        (query_dir / "metadata.json").write_text(
            json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
        )

        # ── 3. reasoning_trace.md — detailed reasoning steps ────────
        reasoning_md_parts = [
            f"# Reasoning Trace\n",
            f"> **Query**: {query}\n",
            f"> **Normalized Query**: {result.get('normalized_query', query)}\n",
            f"> **Query ID**: `{query_id}`\n",
            "",
        ]
        reasoning_history = result.get("reasoning_history", [])
        if reasoning_history:
            reasoning_md_parts.append(format_reasoning_trace(reasoning_history))
        else:
            reasoning_md_parts.append("*No multi-step reasoning (single-shot mode).*\n")

        # Append reflection feedback if present
        reflection = result.get("reflection_feedback", "")
        if reflection:
            reasoning_md_parts += [
                "",
                "## Reflection Feedback",
                "",
                reflection,
                "",
            ]

        (query_dir / "reasoning_trace.md").write_text(
            "\n".join(reasoning_md_parts), encoding="utf-8"
        )

        # ── 4. evidence.json — structured evidence records ──────────
        evidence_data = {
            "retrieved_content": result.get("retrieved_content", []),
            "retrieved_text": result.get("retrieved_text", ""),
            "retrieval_diagnostics": result.get("retrieval_diagnostics", []),
            "web_search_results": result.get("web_search_results", []),
        }
        (query_dir / "evidence.json").write_text(
            json.dumps(evidence_data, indent=2, ensure_ascii=False, default=str),
            encoding="utf-8",
        )

        # ── 5. full_result.pkl — complete pickle for deep inspection
        with open(query_dir / "full_result.pkl", "wb") as f:
            pickle.dump(
                {
                    "query_id": query_id,
                    "timestamp": timestamp,
                    "query": query,
                    "full_result": result,
                    "metadata": metadata,
                    "detailed_reasoning_history": result.get(
                        "detailed_reasoning_history", []
                    ),
                },
                f,
            )

        # ── Update global index ─────────────────────────────────────
        self._append_index_entry(
            {
                "query_id": query_id,
                "timestamp": timestamp.isoformat(),
                "query": query[:100],
                "success": result.get("success", False),
                "iterations": result.get("iterations", 0),
                "mode": result.get("mode", ""),
            }
        )

        print(f"[QueryLogger] Logged query: {query_id}  →  {query_dir}")
        return query_id

    @staticmethod
    def _extract_runtime_metadata(metadata: Dict[str, Any]) -> Dict[str, Any]:
        runtime_keys = (
            "git_sha",
            "model",
            "base_url",
            "doctor_summary",
            "network_notes",
            "suite_name",
        )
        return {
            key: metadata[key]
            for key in runtime_keys
            if str(metadata.get(key, "") or "").strip()
        }

    def save_scorecard_summary(
        self,
        summary: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        timestamp = datetime.now().isoformat()
        payload = {
            "timestamp": timestamp,
            "summary": dict(summary or {}),
            "metadata": dict(metadata or {}),
        }
        scorecard_dir = self.log_dir / "scorecards"
        scorecard_dir.mkdir(parents=True, exist_ok=True)
        timestamped_path = scorecard_dir / (
            f"scorecard_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.json"
        )
        latest_path = self.log_dir / "latest_scorecard.json"
        rendered = json.dumps(payload, indent=2, ensure_ascii=False)
        timestamped_path.write_text(rendered, encoding="utf-8")
        latest_path.write_text(rendered, encoding="utf-8")
        return str(latest_path)

    def get_latest_scorecard_summary(self) -> Optional[Dict[str, Any]]:
        latest_path = self.log_dir / "latest_scorecard.json"
        if not latest_path.exists():
            return None
        return json.loads(latest_path.read_text(encoding="utf-8"))

    @staticmethod
    def _summarize_query_plan(query_plan: Any) -> Dict[str, Any]:
        if not query_plan:
            return {}

        def _read(value: Any, key: str, default: Any = None) -> Any:
            if isinstance(value, dict):
                return value.get(key, default)
            return getattr(value, key, default)

        primary_task = _read(query_plan, "primary_task", {}) or {}
        supporting_tasks = list(_read(query_plan, "supporting_tasks", []) or [])
        answer_contract = _read(query_plan, "answer_contract", {}) or {}
        knowhow_hints = list(_read(query_plan, "knowhow_hints", []) or [])

        summary = {
            "plan_type": str(_read(query_plan, "plan_type", "") or "").strip(),
            "question_type": str(_read(query_plan, "question_type", "") or "").strip(),
            "primary_task_type": str(_read(primary_task, "task_type", "") or "").strip(),
            "supporting_task_types": [
                str(_read(task, "task_type", "") or "").strip()
                for task in supporting_tasks
                if str(_read(task, "task_type", "") or "").strip()
            ],
            "answer_section_order": list(_read(answer_contract, "section_order", []) or []),
            "knowhow_doc_ids": list(_read(query_plan, "knowhow_doc_ids", []) or []),
            "knowhow_hints": [
                {
                    "doc_id": str(_read(hint, "doc_id", "") or "").strip(),
                    "task_id": str(_read(hint, "task_id", "") or "").strip(),
                    "task_type": str(_read(hint, "task_type", "") or "").strip(),
                    "declared_by_skills": [
                        str(value).strip()
                        for value in list(_read(hint, "declared_by_skills", []) or [])
                        if str(value).strip()
                    ],
                }
                for hint in knowhow_hints
                if str(_read(hint, "doc_id", "") or "").strip()
            ],
        }
        return {key: value for key, value in summary.items() if value not in ({}, [], "")}

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def get_query(
        self, query_id: str, detailed: bool = False
    ) -> Optional[Dict[str, Any]]:
        """
        Retrieve a logged query by ID.

        If detailed=True, loads from pickle. Otherwise reads metadata.json.
        """
        query_dir = self.log_dir / query_id

        if not query_dir.is_dir():
            return None

        if detailed:
            pkl = query_dir / "full_result.pkl"
            if pkl.exists():
                with open(pkl, "rb") as f:
                    return pickle.load(f)
            return None

        meta_file = query_dir / "metadata.json"
        if meta_file.exists():
            return json.loads(meta_file.read_text(encoding="utf-8"))
        return None

    def get_query_answer_md(self, query_id: str) -> Optional[str]:
        """Return the rich Markdown answer for a query."""
        md_file = self.log_dir / query_id / "answer.md"
        if md_file.exists():
            return md_file.read_text(encoding="utf-8")
        return None

    def get_query_reasoning_md(self, query_id: str) -> Optional[str]:
        """Return the reasoning trace Markdown for a query."""
        md_file = self.log_dir / query_id / "reasoning_trace.md"
        if md_file.exists():
            return md_file.read_text(encoding="utf-8")
        return None

    def get_query_report_md_path(self, query_id: str) -> Optional[str]:
        """Return the saved Markdown report path for a query if present."""
        md_file = self.log_dir / query_id / "report.md"
        if md_file.exists():
            return str(md_file)
        return None

    def get_recent_queries(self, n: int = 10) -> List[Dict[str, Any]]:
        """Get the N most recent queries from the index."""
        recent = self.index["queries"][-n:]
        recent.reverse()
        return recent

    def search_queries(
        self,
        keyword: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        success_only: bool = False,
    ) -> List[Dict[str, Any]]:
        """Search queries with filters by scanning per-folder metadata."""
        results: List[Dict[str, Any]] = []

        for entry in self.index.get("queries", []):
            qid = entry.get("query_id", "")
            meta_file = self.log_dir / qid / "metadata.json"
            if not meta_file.exists():
                continue

            meta = json.loads(meta_file.read_text(encoding="utf-8"))

            if keyword and keyword.lower() not in meta.get("query", "").lower():
                continue
            if success_only and not meta.get("success", False):
                continue

            entry_time = datetime.fromisoformat(meta.get("timestamp", ""))
            if start_date and entry_time < start_date:
                continue
            if end_date and entry_time > end_date:
                continue

            results.append(meta)

        return results

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def get_statistics(self) -> Dict[str, Any]:
        stats = {
            "total_queries": self.index["total_queries"],
            "successful_queries": 0,
            "failed_queries": 0,
            "avg_iterations": 0,
            "avg_reward": 0,
            "avg_graph_size": 0,
        }

        total_iter = total_reward = total_graph = 0
        count = 0

        for entry in self.index.get("queries", []):
            qid = entry.get("query_id", "")
            meta_file = self.log_dir / qid / "metadata.json"
            if not meta_file.exists():
                continue

            meta = json.loads(meta_file.read_text(encoding="utf-8"))
            count += 1
            if meta.get("success", False):
                stats["successful_queries"] += 1
            else:
                stats["failed_queries"] += 1

            total_iter  += meta.get("iterations", 0)
            total_reward += meta.get("final_reward", 0)
            total_graph  += meta.get("evidence_graph_size", 0)

        if count > 0:
            stats["avg_iterations"]  = total_iter / count
            stats["avg_reward"]      = total_reward / count
            stats["avg_graph_size"]  = total_graph / count

        return stats

    # ------------------------------------------------------------------
    # Export & maintenance
    # ------------------------------------------------------------------

    def export_to_csv(self, output_file: str):
        """Export query history to CSV."""
        import csv

        fieldnames = [
            "query_id", "timestamp", "query", "answer",
            "mode", "iterations", "evidence_graph_size", "final_reward", "success",
        ]

        with open(output_file, "w", newline="", encoding="utf-8") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

            for entry in self.index.get("queries", []):
                qid = entry.get("query_id", "")
                meta_file = self.log_dir / qid / "metadata.json"
                if not meta_file.exists():
                    continue

                meta = json.loads(meta_file.read_text(encoding="utf-8"))
                # Read answer from answer.md for the CSV
                answer_md = self.log_dir / qid / "answer.md"
                answer_text = ""
                if answer_md.exists():
                    answer_text = answer_md.read_text(encoding="utf-8")[:500]

                row = {k: meta.get(k, "") for k in fieldnames}
                row["answer"] = answer_text
                writer.writerow(row)

        print(f"[QueryLogger] Exported to {output_file}")

    def clear_logs(self, confirm: bool = False):
        """Clear all query logs (use with caution!)."""
        if not confirm:
            print("[QueryLogger] Set confirm=True to clear logs")
            return

        import shutil

        for entry in self.index.get("queries", []):
            qid = entry.get("query_id", "")
            qdir = self.log_dir / qid
            if qdir.is_dir():
                shutil.rmtree(qdir)

        self.index = {"total_queries": 0, "queries": []}
        self._save_index()
        print("[QueryLogger] All logs cleared")

    # ------------------------------------------------------------------
    # Reasoning history (backward-compat + enhanced)
    # ------------------------------------------------------------------

    def get_detailed_reasoning_history(
        self, query_id: str
    ) -> Optional[List[Dict[str, Any]]]:
        """Get detailed reasoning history for a specific query."""
        query_dir = self.log_dir / query_id
        pkl = query_dir / "full_result.pkl"
        if not pkl.exists():
            return None

        with open(pkl, "rb") as f:
            data = pickle.load(f)
            return data.get("detailed_reasoning_history", [])

    def print_reasoning_trace(self, query_id: str):
        """Print the Markdown reasoning trace for a query."""
        md = self.get_query_reasoning_md(query_id)
        if md:
            print(md)
        else:
            print(f"No reasoning trace found for {query_id}")


class QuerySession:
    """
    Manages a session of related queries.
    Useful for tracking conversations or research sessions.
    """

    def __init__(
        self, session_id: Optional[str] = None, log_dir: str = "./query_logs"
    ):
        self.logger = QueryLogger(log_dir)
        self.session_id = session_id or f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.queries: List[str] = []
        self.start_time = datetime.now()

    def log_query(self, query: str, result: Dict[str, Any]) -> str:
        metadata = {
            "session_id": self.session_id,
            "query_number": len(self.queries) + 1,
        }
        query_id = self.logger.log_query(query, result, metadata)
        self.queries.append(query_id)
        return query_id

    def get_session_summary(self) -> Dict[str, Any]:
        duration = (datetime.now() - self.start_time).total_seconds()
        return {
            "session_id": self.session_id,
            "start_time": self.start_time.isoformat(),
            "duration_seconds": duration,
            "total_queries": len(self.queries),
            "query_ids": self.queries,
        }
