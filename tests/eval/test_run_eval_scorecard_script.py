from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType


def _load_script_module() -> ModuleType:
    script_path = Path(__file__).resolve().parents[2] / "scripts" / "run_eval_scorecard.py"
    spec = importlib.util.spec_from_file_location("run_eval_scorecard_script", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_run_eval_scorecard_script_delegates_to_cli_scorecard(monkeypatch, tmp_path) -> None:
    module = _load_script_module()
    observed = {}

    def _fake_run_scorecard(**kwargs):
        observed.update(kwargs)
        return 0

    monkeypatch.setattr(module, "_run_scorecard", _fake_run_scorecard)

    exit_code = module.main(
        [
            "--datasets",
            "ade_corpus",
            "ddi_corpus",
            "--task-types",
            "major_adrs",
            "--plan-types",
            "single_task",
            "--max-samples",
            "3",
            "--key-file",
            "navigator_api_keys.json",
            "--maskself",
            "true",
            "--log-dir",
            str(tmp_path / "query_logs"),
            "--json",
            "--min-task-success-rate",
            "0.9",
            "--min-evidence-quality-score",
            "0.8",
        ]
    )

    assert exit_code == 0
    assert observed == {
        "datasets": ["ade_corpus", "ddi_corpus"],
        "task_types": ["major_adrs"],
        "plan_types": ["single_task"],
        "key_file": "navigator_api_keys.json",
        "max_samples": 3,
        "maskself": True,
        "log_dir": str(tmp_path / "query_logs"),
        "json_output": True,
        "min_task_success_rate": 0.9,
        "min_evidence_quality_score": 0.8,
        "min_authority_source_rate": None,
        "min_knowhow_hit_rate": None,
        "min_package_ready_rate": None,
    }
