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
