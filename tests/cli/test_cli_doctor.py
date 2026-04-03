import importlib
import json
import sys
from pathlib import Path

from drugclaw import cli
from drugclaw.cli import _doctor_check_key_file


def test_doctor_accepts_legacy_key_name(tmp_path: Path) -> None:
    key_file = tmp_path / "legacy.json"
    key_file.write_text(
        json.dumps(
            {
                "OPENAI_API_KEY": "legacy-key",
                "base_url": "https://legacy.example.com/v1",
            }
        ),
        encoding="utf-8",
    )

    lines = _doctor_check_key_file(str(key_file))

    assert any("[OK] OPENAI_API_KEY: present" == line for line in lines)


def test_doctor_accepts_new_key_name(tmp_path: Path) -> None:
    key_file = tmp_path / "new.json"
    key_file.write_text(
        json.dumps(
            {
                "api_key": "new-key",
                "base_url": "https://provider.example.com/v1",
                "model": "gemini-3-pro-all",
            }
        ),
        encoding="utf-8",
    )

    lines = _doctor_check_key_file(str(key_file))

    assert any("[OK] OPENAI_API_KEY: present" == line for line in lines)
    assert any("[OK] base_url: https://provider.example.com/v1" == line for line in lines)


def test_doctor_registry_treats_optional_missing_metadata_as_warning(monkeypatch) -> None:
    class _ResourceRegistryStub:
        def summarize_registry(self):
            return {
                "total_resources": 3,
                "enabled_resources": 3,
                "status_counts": {
                    "ready": 1,
                    "degraded": 0,
                    "missing_metadata": 1,
                    "missing_dependency": 0,
                    "disabled": 1,
                },
            }

        def get_all_resources(self):
            return [
                type("Entry", (), {
                    "name": "BindingDB",
                    "category": "dti",
                    "access_mode": "REST_API",
                    "status": "ready",
                    "status_reason": "available",
                    "enabled": True,
                })(),
                type("Entry", (), {
                    "name": "TTD",
                    "category": "dti",
                    "access_mode": "LOCAL_FILE",
                    "status": "missing_metadata",
                    "status_reason": "requires local metadata but no metadata path is configured",
                    "enabled": True,
                })(),
                type("Entry", (), {
                    "name": "PROMISCUOUS 2.0",
                    "category": "dti",
                    "access_mode": "REST_API",
                    "status": "disabled",
                    "status_reason": "not enabled in the runtime skill registry",
                    "enabled": False,
                })(),
            ]

    monkeypatch.setattr(
        cli,
        "_load_registry_for_cli",
        lambda key_file, strict_config=False: (object(), _ResourceRegistryStub()),
    )

    lines = cli._doctor_check_registry("navigator_api_keys.json")

    assert any(line.startswith("[WARN] resource:TTD:") for line in lines)
    assert any(line.startswith("[OK] resource:PROMISCUOUS 2.0:") for line in lines)


def test_doctor_demo_check_uses_runtime_availability_for_local_skills(monkeypatch) -> None:
    class _SkillStub:
        access_mode = "LOCAL_FILE"
        config = {}

        def is_available(self):
            return True

    class _RegistryStub:
        def get_skill(self, name):
            if name == "SIDER":
                return _SkillStub()
            if name == "FAERS":
                skill = type("FAERSSkill", (), {"access_mode": "REST_API", "config": {}, "is_available": lambda self: True})()
                return skill
            if name == "ChEMBL":
                skill = type(
                    "ChEMBLSkill",
                    (),
                    {
                        "access_mode": "CLI",
                        "config": {},
                        "is_available": lambda self: True,
                        "_cli_available": lambda self: False,
                    },
                )()
                return skill
            if name in {"DGIdb", "Open Targets Platform", "DailyMed", "openFDA Human Drug", "MedlinePlus Drug Info"}:
                return type("Skill", (), {"access_mode": "REST_API", "config": {}, "is_available": lambda self: True})()
            return None

    class _ResourceRegistryStub:
        def get_resource(self, name):
            if name == "SIDER":
                return type("Entry", (), {"access_mode": "LOCAL_FILE", "status_reason": "available"})()
            if name == "FAERS":
                return type("Entry", (), {"access_mode": "REST_API", "status_reason": "available"})()
            if name == "ChEMBL":
                return type("Entry", (), {"access_mode": "CLI", "status_reason": "available"})()
            if name in {"DGIdb", "Open Targets Platform", "DailyMed", "openFDA Human Drug", "MedlinePlus Drug Info"}:
                return type("Entry", (), {"access_mode": "REST_API", "status_reason": "available"})()
            return None

    monkeypatch.setattr(
        cli,
        "_load_registry_for_cli",
        lambda key_file, strict_config=True: (_RegistryStub(), _ResourceRegistryStub()),
    )

    lines = cli._doctor_check_presets("navigator_api_keys.json")

    assert any(line.startswith("[OK] demo:adr:") for line in lines)
    assert any("SIDER=LOCAL_FILE:available" in line for line in lines)


def test_doctor_demo_check_allows_partial_preset_availability_when_one_resource_is_usable(monkeypatch) -> None:
    class _RegistryStub:
        def get_skill(self, name):
            if name == "SIDER":
                return type(
                    "SIDERSkill",
                    (),
                    {"access_mode": "LOCAL_FILE", "config": {}, "is_available": lambda self: False},
                )()
            if name == "FAERS":
                return type(
                    "FAERSSkill",
                    (),
                    {"access_mode": "REST_API", "config": {}, "is_available": lambda self: True},
                )()
            if name in {"ChEMBL", "DGIdb", "Open Targets Platform", "DailyMed", "openFDA Human Drug", "MedlinePlus Drug Info"}:
                return type("Skill", (), {"access_mode": "REST_API", "config": {}, "is_available": lambda self: True})()
            return None

    class _ResourceRegistryStub:
        def get_resource(self, name):
            if name == "SIDER":
                return type(
                    "Entry",
                    (),
                    {
                        "access_mode": "LOCAL_FILE",
                        "status_reason": "requires local metadata but no metadata path is configured",
                    },
                )()
            if name == "FAERS":
                return type("Entry", (), {"access_mode": "REST_API", "status_reason": "available"})()
            if name in {"ChEMBL", "DGIdb", "Open Targets Platform", "DailyMed", "openFDA Human Drug", "MedlinePlus Drug Info"}:
                return type("Entry", (), {"access_mode": "REST_API", "status_reason": "available"})()
            return None

    monkeypatch.setattr(
        cli,
        "_load_registry_for_cli",
        lambda key_file, strict_config=True: (_RegistryStub(), _ResourceRegistryStub()),
    )

    lines = cli._doctor_check_presets("navigator_api_keys.json")

    assert any(
        line.startswith("[OK] demo:adr:")
        and "SIDER=LOCAL_FILE:requires local metadata but no metadata path is configured" in line
        and "FAERS=REST_API:available" in line
        for line in lines
    )


def test_doctor_ignores_warnings_when_computing_exit_code(monkeypatch, capsys) -> None:
    monkeypatch.setattr(cli, "_doctor_check_key_file", lambda key_file: ["[OK] key_file: ok"])
    monkeypatch.setattr(cli, "_doctor_check_imports", lambda: ["[OK] openai: importable"])
    monkeypatch.setattr(cli, "_doctor_check_registry", lambda key_file: ["[WARN] resource:TTD: missing local metadata"])
    monkeypatch.setattr(cli, "_doctor_check_presets", lambda key_file: ["[OK] demo:dti: available"])
    monkeypatch.setattr(cli, "_doctor_check_install_hint", lambda: ["[OK] cli_command: command found on PATH"])
    monkeypatch.setattr(cli, "_doctor_check_git_safety", lambda: ["[OK] tracked_key_file: not tracked"])

    exit_code = cli._run_doctor("navigator_api_keys.json")

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "[WARN] resource:TTD: missing local metadata" in captured.out
    assert "Doctor result: setup looks usable." in captured.out


def test_doctor_install_hint_accepts_current_python_environment(tmp_path: Path, monkeypatch) -> None:
    venv_bin = tmp_path / "bin"
    venv_bin.mkdir()
    (venv_bin / "python").write_text("", encoding="utf-8")
    (venv_bin / "drugclaw").write_text("", encoding="utf-8")

    monkeypatch.setattr(cli.sys, "executable", str(venv_bin / "python"))
    monkeypatch.setenv("PATH", "")

    lines = cli._doctor_check_install_hint()

    assert lines == ["[OK] cli_command: command available for the current Python environment"]


def test_doctor_install_hint_uses_symlink_path_parent(tmp_path: Path, monkeypatch) -> None:
    venv_bin = tmp_path / "bin"
    venv_bin.mkdir()
    target_python = venv_bin / "python3"
    target_python.write_text("", encoding="utf-8")
    (venv_bin / "python").symlink_to(target_python.name)
    (venv_bin / "drugclaw").write_text("", encoding="utf-8")

    monkeypatch.setattr(cli.sys, "executable", str(venv_bin / "python"))
    monkeypatch.setenv("PATH", "")

    lines = cli._doctor_check_install_hint()

    assert lines == ["[OK] cli_command: command available for the current Python environment"]


def test_cli_module_can_reload_without_main_system(monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "drugclaw.main_system", None)

    reloaded = importlib.reload(cli)

    try:
        parser = reloaded.build_parser()
        args = parser.parse_args(["doctor"])
        assert args.key_file == "navigator_api_keys.json"
    finally:
        importlib.reload(cli)


def test_cli_help_prefers_navigator_key_filename() -> None:
    parser = cli.build_parser()
    run_help = parser._subparsers._group_actions[0].choices["run"].format_help()
    doctor_help = parser._subparsers._group_actions[0].choices["doctor"].format_help()
    list_help = parser._subparsers._group_actions[0].choices["list"].format_help()

    assert "navigator_api_keys.json" in run_help
    assert "navigator_api_keys.json" in doctor_help
    normalized_list_help = " ".join(list_help.split())
    assert "Path to navigator_api_keys.json. Use --key-file api_keys.json to override." in normalized_list_help
    assert "Path to api_keys.json or navigator_api_keys.json." not in list_help


def test_doctor_and_list_do_not_require_main_system(monkeypatch, capsys) -> None:
    monkeypatch.setitem(sys.modules, "drugclaw.main_system", None)
    reloaded = importlib.reload(cli)

    try:
        monkeypatch.setattr(reloaded, "_doctor_check_key_file", lambda key_file: ["[OK] key_file: ok"])
        monkeypatch.setattr(reloaded, "_doctor_check_imports", lambda: ["[FAIL] langgraph: import failed (No module named 'langgraph')"])
        monkeypatch.setattr(reloaded, "_doctor_check_registry", lambda key_file: ["[OK] registry: skipped"])
        monkeypatch.setattr(reloaded, "_doctor_check_presets", lambda key_file: ["[OK] demo:label: available"])
        monkeypatch.setattr(reloaded, "_doctor_check_install_hint", lambda: ["[OK] cli_command: command found on PATH"])
        monkeypatch.setattr(reloaded, "_doctor_check_git_safety", lambda: ["[OK] tracked_key_file: not tracked"])
        assert reloaded._run_doctor("navigator_api_keys.json") == 1

        class _ResourceRegistryStub:
            def summarize_registry(self):
                return {
                    "total_resources": 1,
                    "enabled_resources": 1,
                    "status_counts": {
                        "ready": 1,
                        "degraded": 0,
                        "missing_metadata": 0,
                        "missing_dependency": 0,
                        "disabled": 0,
                    },
                }

            def get_all_resources(self):
                return []

        monkeypatch.setattr(
            reloaded,
            "_load_registry_for_cli",
            lambda key_file, strict_config=False: (object(), _ResourceRegistryStub()),
        )
        assert reloaded._run_list("navigator_api_keys.json") == 0
    finally:
        importlib.reload(cli)
