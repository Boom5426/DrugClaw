import json

from drugclaw import cli


def test_run_query_does_not_print_answer_twice(monkeypatch, capsys) -> None:
    class FakeSystem:
        def __init__(self, config):
            self.config = config

        def query(
            self,
            query,
            thinking_mode,
            resource_filter,
            verbose=True,
            save_md_report=False,
        ):
            assert verbose is False
            assert save_md_report is False
            return {
                "answer": "only-once",
                "formatted_answer": "only-once",
                "success": True,
            }

    monkeypatch.setattr(cli, "_build_system", lambda key_file: FakeSystem(object()))

    exit_code = cli._run_query(
        query="demo",
        thinking_mode="simple",
        key_file="navigator_api_keys.json",
        resource_filter=["DailyMed"],
    )

    captured = capsys.readouterr()

    assert exit_code == 0
    assert captured.out.count("only-once") == 1


def test_run_query_can_print_structured_evidence_summary(monkeypatch, capsys) -> None:
    class FakeSystem:
        def __init__(self, config):
            self.config = config

        def query(
            self,
            query,
            thinking_mode,
            resource_filter,
            verbose=True,
            save_md_report=False,
        ):
            assert verbose is False
            assert save_md_report is False
            return {
                "answer": "answer-once",
                "formatted_answer": "answer-once",
                "success": True,
                "final_answer_structured": {
                    "summary_confidence": 0.75,
                    "task_type": "mechanism",
                    "final_outcome": "partial_with_weak_support",
                    "diagnostics": {
                        "strong_record_count": 1,
                        "weak_support_count": 0,
                    },
                    "key_claims": [
                        {
                            "claim": "Imatinib targets ABL1.",
                            "confidence": 0.75,
                            "evidence_ids": ["E1", "E2"],
                        }
                    ],
                    "warnings": ["Evidence conflict detected for claim: Imatinib targets ABL1."],
                    "limitations": ["Claim relies on a single source: Imatinib targets ABL1."],
                },
            }

    monkeypatch.setattr(cli, "_build_system", lambda key_file: FakeSystem(object()))

    exit_code = cli._run_query(
        query="demo",
        thinking_mode="simple",
        key_file="navigator_api_keys.json",
        resource_filter=["BindingDB"],
        show_evidence=True,
    )

    captured = capsys.readouterr()

    assert exit_code == 0
    assert "summary_confidence=0.75" in captured.out
    assert "task_type=mechanism" in captured.out
    assert "final_outcome=partial_with_weak_support" in captured.out
    assert "Imatinib targets ABL1." in captured.out


def test_run_query_can_print_plan_and_claim_summaries(monkeypatch, capsys) -> None:
    class FakeSystem:
        def __init__(self, config):
            self.config = config

        def query(
            self,
            query,
            thinking_mode,
            resource_filter,
            verbose=True,
            save_md_report=False,
        ):
            assert verbose is False
            assert save_md_report is False
            return {
                "answer": "answer-once",
                "formatted_answer": "answer-once",
                "success": True,
                "query_plan": {
                    "plan_type": "composite_query",
                    "question_type": "target_lookup",
                    "primary_task": {"task_type": "direct_targets"},
                    "supporting_tasks": [{"task_type": "mechanism_of_action"}],
                    "preferred_skills": ["BindingDB", "ChEMBL"],
                    "requires_graph_reasoning": False,
                },
                "claim_assessments": [
                    {
                        "claim": "Imatinib targets ABL1.",
                        "verdict": "supported",
                        "confidence": 0.88,
                        "supporting_evidence_ids": ["E1"],
                        "contradicting_evidence_ids": [],
                    }
                ],
                "graph_decision_reason": "skip:planner did not recommend graph reasoning",
            }

    monkeypatch.setattr(cli, "_build_system", lambda key_file: FakeSystem(object()))

    exit_code = cli._run_query(
        query="demo",
        thinking_mode="simple",
        key_file="navigator_api_keys.json",
        resource_filter=["BindingDB"],
        show_plan=True,
        show_claims=True,
    )

    captured = capsys.readouterr()

    assert exit_code == 0
    assert captured.out.count("answer-once") == 1
    assert "plan_type=composite_query" in captured.out
    assert "question_type=target_lookup" in captured.out
    assert "primary_task=direct_targets" in captured.out
    assert "supporting_tasks=mechanism_of_action" in captured.out
    assert "preferred_skills=BindingDB, ChEMBL" in captured.out
    assert "verdict=supported" in captured.out
    assert "graph_decision_reason=skip:planner did not recommend graph reasoning" in captured.out


def test_run_query_can_enable_agent_debug_output(monkeypatch, capsys) -> None:
    class FakeSystem:
        def __init__(self, config):
            self.config = config

        def query(
            self,
            query,
            thinking_mode,
            resource_filter,
            verbose=True,
            save_md_report=False,
        ):
            assert verbose is True
            assert save_md_report is False
            print("[Retriever Agent] debug log")
            return {
                "answer": "answer-once",
                "formatted_answer": "answer-once",
                "success": True,
            }

    monkeypatch.setattr(cli, "_build_system", lambda key_file: FakeSystem(object()))

    exit_code = cli._run_query(
        query="demo",
        thinking_mode="simple",
        key_file="navigator_api_keys.json",
        resource_filter=["BindingDB"],
        debug_agents=True,
    )

    captured = capsys.readouterr()

    assert exit_code == 0
    assert "[Retriever Agent] debug log" in captured.out


def test_run_query_can_print_saved_md_report_path(monkeypatch, capsys) -> None:
    class FakeSystem:
        def __init__(self, config):
            self.config = config

        def query(
            self,
            query,
            thinking_mode,
            resource_filter,
            verbose=True,
            save_md_report=False,
        ):
            assert verbose is False
            assert save_md_report is True
            return {
                "answer": "answer-once",
                "formatted_answer": "answer-once",
                "success": True,
                "md_report_path": "query_logs/query_1/report.md",
            }

    monkeypatch.setattr(cli, "_build_system", lambda key_file: FakeSystem(object()))

    exit_code = cli._run_query(
        query="demo",
        thinking_mode="simple",
        key_file="navigator_api_keys.json",
        resource_filter=["BindingDB"],
        save_md_report=True,
    )

    captured = capsys.readouterr()

    assert exit_code == 0
    assert "Markdown report saved to query_logs/query_1/report.md" in captured.out


def test_run_query_omits_front_loaded_query_banner_by_default(monkeypatch, capsys) -> None:
    class FakeSystem:
        def __init__(self, config):
            self.config = config

        def query(
            self,
            query,
            thinking_mode,
            resource_filter,
            verbose=True,
            save_md_report=False,
        ):
            assert verbose is False
            return {
                "answer": "answer-once",
                "formatted_answer": "# DrugClaw Query Report\n\n## Answer\n\nanswer-once",
                "success": True,
            }

    monkeypatch.setattr(cli, "_build_system", lambda key_file: FakeSystem(object()))

    exit_code = cli._run_query(
        query="What are the known drug targets and mechanism of action of imatinib?",
        thinking_mode="simple",
        key_file="navigator_api_keys.json",
        resource_filter=None,
    )

    captured = capsys.readouterr()

    assert exit_code == 0
    assert "FINAL ANSWER" in captured.out
    assert "QUERY [simple]" not in captured.out


def test_run_parser_accepts_save_md_report_flag() -> None:
    parser = cli.build_parser()

    args = parser.parse_args(
        ["run", "--query", "What does imatinib target?", "--save-md-report"]
    )

    assert args.save_md_report is True


def test_scorecard_parser_accepts_filters_and_json_output() -> None:
    parser = cli.build_parser()

    args = parser.parse_args(
        [
            "scorecard",
            "--datasets",
            "ade_corpus",
            "ddi_corpus",
            "--task-types",
            "major_adrs",
            "--json",
            "--log-dir",
            "tmp-query-logs",
        ]
    )

    assert args.command == "scorecard"
    assert args.datasets == ["ade_corpus", "ddi_corpus"]
    assert args.task_types == ["major_adrs"]
    assert args.json is True
    assert args.log_dir == "tmp-query-logs"


def test_scorecard_parser_accepts_release_gate_thresholds() -> None:
    parser = cli.build_parser()

    args = parser.parse_args(
        [
            "scorecard",
            "--min-task-success-rate",
            "0.9",
            "--min-evidence-quality-score",
            "0.8",
            "--min-authority-source-rate",
            "0.7",
            "--min-knowhow-hit-rate",
            "0.6",
            "--min-package-ready-rate",
            "0.5",
        ]
    )

    assert args.min_task_success_rate == 0.9
    assert args.min_evidence_quality_score == 0.8
    assert args.min_authority_source_rate == 0.7
    assert args.min_knowhow_hit_rate == 0.6
    assert args.min_package_ready_rate == 0.5


def test_run_scorecard_prints_summary_and_persists_logger_snapshot(monkeypatch, capsys, tmp_path) -> None:
    from drugclaw.eval_models import EvalRunSummary

    class _Summary:
        def __init__(self):
            self.total_cases = 2
            self.completed_cases = 2
            self.failed_cases = 0
            self.task_success_rate = 1.0
            self.evidence_quality_score = 0.8
            self.authority_source_rate = 0.5
            self.knowhow_hit_rate = 1.0
            self.package_ready_rate = 0.25
            self.scores = []
            self.dataset_results = {
                "ade_corpus": {"accuracy": 0.8, "total": 5, "correct": 4},
                "ddi_corpus": {"accuracy": 0.6, "total": 5, "correct": 3},
            }
            self.dataset_breakdown = {
                "ade_corpus": {
                    "total_cases": 1,
                    "task_success_rate": 1.0,
                    "evidence_quality_score": 0.8,
                }
            }
            self.task_type_breakdown = {
                "major_adrs": {
                    "total_cases": 1,
                    "task_success_rate": 1.0,
                    "evidence_quality_score": 0.8,
                }
            }

        def to_dict(self):
            return {
                "total_cases": self.total_cases,
                "completed_cases": self.completed_cases,
                "failed_cases": self.failed_cases,
                "task_success_rate": self.task_success_rate,
                "evidence_quality_score": self.evidence_quality_score,
                "authority_source_rate": self.authority_source_rate,
                "knowhow_hit_rate": self.knowhow_hit_rate,
                "package_ready_rate": self.package_ready_rate,
                "scores": [],
                "dataset_results": self.dataset_results,
                "dataset_breakdown": self.dataset_breakdown,
                "task_type_breakdown": self.task_type_breakdown,
            }

    class _LoggerStub:
        def __init__(self, log_dir):
            self.log_dir = log_dir
            self.saved = None

        def save_scorecard_summary(self, summary, metadata=None):
            self.saved = {"summary": summary, "metadata": metadata}
            return str(tmp_path / "query_logs" / "latest_scorecard.json")

    logger = _LoggerStub(str(tmp_path / "query_logs"))
    monkeypatch.setattr(cli, "run_self_bench", lambda **kwargs: _Summary())
    monkeypatch.setattr(cli, "QueryLogger", lambda log_dir="": logger)

    exit_code = cli._run_scorecard(
        datasets=["ade_corpus", "ddi_corpus"],
        task_types=["major_adrs"],
        plan_types=None,
        key_file="navigator_api_keys.json",
        max_samples=3,
        maskself=True,
        log_dir=str(tmp_path / "query_logs"),
        json_output=False,
        min_task_success_rate=None,
        min_evidence_quality_score=None,
        min_authority_source_rate=None,
        min_knowhow_hit_rate=None,
        min_package_ready_rate=None,
    )

    captured = capsys.readouterr()

    assert exit_code == 0
    assert "task_success_rate=1.00" in captured.out
    assert "evidence_quality_score=0.80" in captured.out
    assert "authority_source_rate=0.50" in captured.out
    assert "[By task type]" in captured.out
    assert "- major_adrs: success=1.00 evidence=0.80 total_cases=1" in captured.out
    assert "[By dataset]" in captured.out
    assert "- ade_corpus: success=1.00 evidence=0.80 total_cases=1" in captured.out
    assert "latest_scorecard.json" in captured.out
    assert logger.saved is not None
    assert logger.saved["summary"]["release_gate"]["passed"] is True
    assert logger.saved["summary"]["release_gate"]["failures"] == []
    assert logger.saved["summary"]["release_gate"]["by_task_type"]["major_adrs"]["passed"] is True
    assert logger.saved["summary"]["release_gate"]["by_dataset"]["ade_corpus"]["passed"] is True
    assert logger.saved["metadata"]["datasets"] == ["ade_corpus", "ddi_corpus"]
    assert logger.saved["metadata"]["task_types"] == ["major_adrs"]


def test_run_scorecard_fails_when_release_gate_threshold_is_not_met(monkeypatch, capsys, tmp_path) -> None:
    class _Summary:
        total_cases = 1
        completed_cases = 1
        failed_cases = 0
        task_success_rate = 0.85
        evidence_quality_score = 0.8
        authority_source_rate = 0.7
        knowhow_hit_rate = 1.0
        package_ready_rate = 0.5
        scores = []
        dataset_results = {"ade_corpus": {"accuracy": 0.8, "total": 5, "correct": 4}}
        dataset_breakdown = {
            "ade_corpus": {
                "total_cases": 1,
                "task_success_rate": 0.85,
                "evidence_quality_score": 0.8,
            }
        }
        task_type_breakdown = {
            "major_adrs": {
                "total_cases": 1,
                "task_success_rate": 0.85,
                "evidence_quality_score": 0.8,
            }
        }

        def to_dict(self):
            return {
                "total_cases": self.total_cases,
                "completed_cases": self.completed_cases,
                "failed_cases": self.failed_cases,
                "task_success_rate": self.task_success_rate,
                "evidence_quality_score": self.evidence_quality_score,
                "authority_source_rate": self.authority_source_rate,
                "knowhow_hit_rate": self.knowhow_hit_rate,
                "package_ready_rate": self.package_ready_rate,
                "scores": [],
                "dataset_results": self.dataset_results,
                "dataset_breakdown": self.dataset_breakdown,
                "task_type_breakdown": self.task_type_breakdown,
            }

    class _LoggerStub:
        def __init__(self, log_dir):
            self.log_dir = log_dir

        def save_scorecard_summary(self, summary, metadata=None):
            return str(tmp_path / "query_logs" / "latest_scorecard.json")

    monkeypatch.setattr(cli, "run_self_bench", lambda **kwargs: _Summary())
    monkeypatch.setattr(cli, "QueryLogger", lambda log_dir="": _LoggerStub(log_dir))

    exit_code = cli._run_scorecard(
        datasets=["ade_corpus"],
        task_types=["major_adrs"],
        plan_types=None,
        key_file="navigator_api_keys.json",
        max_samples=3,
        maskself=True,
        log_dir=str(tmp_path / "query_logs"),
        json_output=False,
        min_task_success_rate=0.9,
        min_evidence_quality_score=None,
        min_authority_source_rate=None,
        min_knowhow_hit_rate=None,
        min_package_ready_rate=None,
    )

    captured = capsys.readouterr()

    assert exit_code == 1
    assert "Release gate failures:" in captured.out
    assert "task_success_rate=0.85 < 0.90" in captured.out


def test_run_scorecard_json_output_includes_release_gate_payload(monkeypatch, capsys, tmp_path) -> None:
    class _Summary:
        total_cases = 1
        completed_cases = 1
        failed_cases = 0
        task_success_rate = 0.85
        evidence_quality_score = 0.8
        authority_source_rate = 0.7
        knowhow_hit_rate = 1.0
        package_ready_rate = 0.5
        scores = []
        dataset_results = {"ade_corpus": {"accuracy": 0.8, "total": 5, "correct": 4}}
        dataset_breakdown = {
            "ade_corpus": {
                "total_cases": 1,
                "task_success_rate": 0.85,
                "evidence_quality_score": 0.8,
            }
        }
        task_type_breakdown = {
            "major_adrs": {
                "total_cases": 1,
                "task_success_rate": 0.85,
                "evidence_quality_score": 0.8,
            }
        }

        def to_dict(self):
            return {
                "total_cases": self.total_cases,
                "completed_cases": self.completed_cases,
                "failed_cases": self.failed_cases,
                "task_success_rate": self.task_success_rate,
                "evidence_quality_score": self.evidence_quality_score,
                "authority_source_rate": self.authority_source_rate,
                "knowhow_hit_rate": self.knowhow_hit_rate,
                "package_ready_rate": self.package_ready_rate,
                "scores": [],
                "dataset_results": self.dataset_results,
                "dataset_breakdown": self.dataset_breakdown,
                "task_type_breakdown": self.task_type_breakdown,
            }

    class _LoggerStub:
        def __init__(self, log_dir):
            self.log_dir = log_dir

        def save_scorecard_summary(self, summary, metadata=None):
            return str(tmp_path / "query_logs" / "latest_scorecard.json")

    monkeypatch.setattr(cli, "run_self_bench", lambda **kwargs: _Summary())
    monkeypatch.setattr(cli, "QueryLogger", lambda log_dir="": _LoggerStub(log_dir))

    exit_code = cli._run_scorecard(
        datasets=["ade_corpus"],
        task_types=["major_adrs"],
        plan_types=None,
        key_file="navigator_api_keys.json",
        max_samples=3,
        maskself=True,
        log_dir=str(tmp_path / "query_logs"),
        json_output=True,
        min_task_success_rate=0.9,
        min_evidence_quality_score=None,
        min_authority_source_rate=None,
        min_knowhow_hit_rate=None,
        min_package_ready_rate=None,
    )

    captured = capsys.readouterr()
    rendered = captured.out.split("\nScorecard summary saved to ", 1)[0]
    payload = json.loads(rendered)

    assert exit_code == 1
    assert payload["release_gate"]["passed"] is False
    assert payload["release_gate"]["failures"] == ["task_success_rate=0.85 < 0.90"]
    assert payload["release_gate"]["thresholds"]["min_task_success_rate"] == 0.9
    assert payload["release_gate"]["by_dataset"]["ade_corpus"]["passed"] is False
    assert payload["release_gate"]["by_dataset"]["ade_corpus"]["failures"] == [
        "task_success_rate=0.85 < 0.90"
    ]
