import json
from pathlib import Path

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
