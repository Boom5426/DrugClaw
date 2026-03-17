"""
Query Logger - Folder-based per-query storage with Markdown reports.

Each query is stored in its own directory:

    query_logs/
    ├── query_index.json                        # Global index
    └── query_20260317_120000_123456/           # One folder per query
        ├── answer.md                           # Rich Markdown answer card
        ├── metadata.json                       # Query metadata + metrics
        ├── reasoning_trace.md                  # Step-by-step reasoning
        ├── evidence.json                       # Structured evidence records
        └── full_result.pkl                     # Complete pickle dump
"""
from __future__ import annotations

import json
import os
import pickle
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

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
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Global index for fast lookup
        self.index_file = self.log_dir / "query_index.json"
        self._load_or_create_index()

    # ------------------------------------------------------------------
    # Index management
    # ------------------------------------------------------------------

    def _load_or_create_index(self):
        if self.index_file.exists():
            with open(self.index_file, "r") as f:
                self.index = json.load(f)
        else:
            self.index = {"total_queries": 0, "queries": []}
            self._save_index()

    def _save_index(self):
        with open(self.index_file, "w") as f:
            json.dump(self.index, indent=2, fp=f)

    # ------------------------------------------------------------------
    # Core logging
    # ------------------------------------------------------------------

    def log_query(
        self,
        query: str,
        result: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None,
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

        # ── 2. metadata.json — query metadata + metrics ─────────────
        meta = {
            "query_id": query_id,
            "timestamp": timestamp.isoformat(),
            "query": query,
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
        (query_dir / "metadata.json").write_text(
            json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
        )

        # ── 3. reasoning_trace.md — detailed reasoning steps ────────
        reasoning_md_parts = [
            f"# Reasoning Trace\n",
            f"> **Query**: {query}\n",
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
        self.index["total_queries"] += 1
        self.index["queries"].append(
            {
                "query_id": query_id,
                "timestamp": timestamp.isoformat(),
                "query": query[:100],
                "success": result.get("success", False),
                "iterations": result.get("iterations", 0),
                "mode": result.get("mode", ""),
            }
        )
        self._save_index()

        print(f"[QueryLogger] Logged query: {query_id}  →  {query_dir}")
        return query_id

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
