from __future__ import annotations

from skills.dti.chembl.chembl_skill import ChEMBLSkill


def test_chembl_select_ranked_activities_prefers_specific_potent_targets_and_dedupes() -> None:
    activities = [
        {
            "target_pref_name": "Platelet-derived growth factor receptor",
            "standard_type": "IC50",
            "standard_value": "25",
        },
        {
            "target_pref_name": "Platelet-derived growth factor receptor alpha",
            "standard_type": "IC50",
            "standard_value": "85",
        },
        {
            "target_pref_name": "ABL1",
            "standard_type": "Ki",
            "standard_value": "21",
        },
        {
            "target_pref_name": "ABL1",
            "standard_type": "IC50",
            "standard_value": "75",
        },
        {
            "target_pref_name": "Receptor-type tyrosine-protein kinase FLT3",
            "standard_type": "IC50",
            "standard_value": "120",
        },
    ]

    selected = ChEMBLSkill._select_ranked_activities(activities, limit=3)

    assert [activity["target_pref_name"] for activity in selected] == [
        "ABL1",
        "Platelet-derived growth factor receptor alpha",
        "Receptor-type tyrosine-protein kinase FLT3",
    ]

