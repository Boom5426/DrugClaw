from __future__ import annotations

from skills.dti.ttd.ttd_skill import TTDSkill


def test_ttd_is_unavailable_without_configured_file() -> None:
    skill = TTDSkill({"drug_target_tsv": ""})

    assert skill.is_available() is False
