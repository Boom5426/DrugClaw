from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
LEGACY_REPO_PATH = "/blue/qsong1/wang.qing/AgentLLM/DrugClaw"


def test_root_does_not_expose_legacy_helper_files() -> None:
    assert not (ROOT / "example_usage.py").exists()
    assert not (ROOT / "run_minimal.py").exists()
    assert not (ROOT / "get_reason_detail.py").exists()
    assert not (ROOT / "query_teamplate.py").exists()
    assert (ROOT / "examples").is_dir()
    assert (ROOT / "scripts" / "legacy").is_dir()


def test_readmes_do_not_contain_machine_specific_cd() -> None:
    assert "cd /data/boom/Agent/DrugClaw" not in (
        ROOT / "README.md"
    ).read_text(encoding="utf-8")
    assert "cd /data/boom/Agent/DrugClaw" not in (
        ROOT / "README_CN.md"
    ).read_text(encoding="utf-8")


def test_skill_examples_do_not_contain_legacy_machine_specific_repo_paths() -> None:
    offenders = []
    for path in sorted((ROOT / "skills").rglob("example.py")):
        content = path.read_text(encoding="utf-8")
        if LEGACY_REPO_PATH in content:
            offenders.append(path.relative_to(ROOT).as_posix())

    assert offenders == []


def test_skill_docs_do_not_contain_legacy_machine_specific_repo_paths() -> None:
    offenders = []
    for path in sorted((ROOT / "skills").rglob("SKILL.md")):
        content = path.read_text(encoding="utf-8")
        if LEGACY_REPO_PATH in content:
            offenders.append(path.relative_to(ROOT).as_posix())

    assert offenders == []


def test_requirements_txt_exists_and_readmes_document_one_shot_install() -> None:
    requirements = ROOT / "requirements.txt"
    assert requirements.exists()

    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    readme_cn = (ROOT / "README_CN.md").read_text(encoding="utf-8")

    assert "python -m pip install --no-build-isolation -r requirements.txt" in readme
    assert "python -m pip install --no-build-isolation -r requirements.txt" in readme_cn
