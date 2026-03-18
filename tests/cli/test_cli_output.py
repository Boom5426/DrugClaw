from drugclaw import cli


def test_run_query_does_not_print_answer_twice(monkeypatch, capsys) -> None:
    class FakeSystem:
        def __init__(self, config):
            self.config = config

        def query(self, query, thinking_mode, resource_filter):
            print("only-once")
            return {"answer": "only-once", "success": True}

    monkeypatch.setattr(cli, "DrugClawSystem", FakeSystem)
    monkeypatch.setattr(cli, "Config", lambda key_file: object())

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

        def query(self, query, thinking_mode, resource_filter):
            print("answer-once")
            return {
                "answer": "answer-once",
                "success": True,
                "final_answer_structured": {
                    "summary_confidence": 0.75,
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

    monkeypatch.setattr(cli, "DrugClawSystem", FakeSystem)
    monkeypatch.setattr(cli, "Config", lambda key_file: object())

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
    assert "Imatinib targets ABL1." in captured.out


def test_run_query_can_print_plan_and_claim_summaries(monkeypatch, capsys) -> None:
    class FakeSystem:
        def __init__(self, config):
            self.config = config

        def query(self, query, thinking_mode, resource_filter):
            print("answer-once")
            return {
                "answer": "answer-once",
                "success": True,
                "query_plan": {
                    "question_type": "target_lookup",
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

    monkeypatch.setattr(cli, "DrugClawSystem", FakeSystem)
    monkeypatch.setattr(cli, "Config", lambda key_file: object())

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
    assert "question_type=target_lookup" in captured.out
    assert "preferred_skills=BindingDB, ChEMBL" in captured.out
    assert "verdict=supported" in captured.out
    assert "graph_decision_reason=skip:planner did not recommend graph reasoning" in captured.out
