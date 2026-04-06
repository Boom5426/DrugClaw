from __future__ import annotations

import json

from skills.drug_labeling.openfda.openfda_skill import OpenFDASkill


class _FakeHTTPResponse:
    def __init__(self, payload: dict):
        self._payload = json.dumps(payload).encode()

    def read(self) -> bytes:
        return self._payload

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None


def test_openfda_emits_identity_metadata_for_exact_and_combination_labels(monkeypatch) -> None:
    payload = {
        "results": [
            {
                "openfda": {
                    "brand_name": ["Metformin Hydrochloride"],
                    "generic_name": ["metformin hydrochloride"],
                    "product_type": ["HUMAN PRESCRIPTION DRUG"],
                },
                "indications_and_usage": [
                    "Used as an adjunct to diet and exercise to improve glycemic control in type 2 diabetes mellitus."
                ],
            },
            {
                "openfda": {
                    "brand_name": ["ZITUVIMET"],
                    "generic_name": ["sitagliptin and metformin hydrochloride"],
                    "product_type": ["HUMAN PRESCRIPTION DRUG"],
                },
                "indications_and_usage": [
                    "ZITUVIMET is indicated as an adjunct to diet and exercise to improve glycemic control in adults with type 2 diabetes mellitus."
                ],
            },
            {
                "openfda": {
                    "brand_name": ["Glipizide and Metformin Hydrochloride"],
                    "generic_name": ["glipizide and metformin hydrochloride"],
                    "product_type": ["HUMAN PRESCRIPTION DRUG"],
                },
                "warnings": [
                    "Glipizide and metformin hydrochloride tablets carry hypoglycemia and lactic acidosis warnings."
                ],
            },
        ]
    }

    monkeypatch.setattr(
        "urllib.request.urlopen",
        lambda *args, **kwargs: _FakeHTTPResponse(payload),
    )

    skill = OpenFDASkill()
    results = skill.retrieve({"drug": ["metformin"]}, max_results=6)

    assert results

    metformin_result = next(
        result for result in results
        if result.source_entity == "Metformin Hydrochloride"
    )
    assert metformin_result.metadata["queried_drug"] == "metformin"
    assert metformin_result.metadata["brand_name"] == "Metformin Hydrochloride"
    assert metformin_result.metadata["generic_names"] == ["metformin hydrochloride"]
    assert metformin_result.metadata["is_combination_product"] is False

    zituvimet_result = next(
        result for result in results
        if result.source_entity == "ZITUVIMET"
    )
    assert zituvimet_result.metadata["queried_drug"] == "metformin"
    assert zituvimet_result.metadata["brand_name"] == "ZITUVIMET"
    assert zituvimet_result.metadata["generic_names"] == [
        "sitagliptin and metformin hydrochloride"
    ]
    assert zituvimet_result.metadata["is_combination_product"] is True
    assert zituvimet_result.metadata["openfda_product_type"] == "HUMAN PRESCRIPTION DRUG"


def test_openfda_emits_contraindication_and_special_population_sections(monkeypatch) -> None:
    payload = {
        "results": [
            {
                "openfda": {
                    "brand_name": ["Metformin Hydrochloride"],
                    "generic_name": ["metformin hydrochloride"],
                    "product_type": ["HUMAN PRESCRIPTION DRUG"],
                },
                "contraindications": [
                    "Metformin is contraindicated in patients with severe renal impairment."
                ],
                "use_in_specific_populations": [
                    "Assess renal function more frequently in older adults and other at-risk populations."
                ],
                "drug_interactions": [
                    "Carbonic anhydrase inhibitors may increase the risk of lactic acidosis; monitor closely."
                ],
            }
        ]
    }

    monkeypatch.setattr(
        "urllib.request.urlopen",
        lambda *args, **kwargs: _FakeHTTPResponse(payload),
    )

    skill = OpenFDASkill()
    results = skill.retrieve({"drug": ["metformin"]}, max_results=6)

    relationships = {result.relationship for result in results}
    fields = {result.metadata["field"] for result in results}

    assert "has_contraindication" in relationships
    assert "use_in_special_population" in relationships
    assert "interacts_with" in relationships
    assert "contraindications" in fields
    assert "use_in_specific_populations" in fields


def test_openfda_truncates_long_sections_at_word_boundaries(monkeypatch) -> None:
    long_warning = ("word " * 99) + "disintegrate tablets should dissolve on the tongue."
    payload = {
        "results": [
            {
                "openfda": {
                    "brand_name": ["Clozapine ODT"],
                    "generic_name": ["clozapine"],
                    "product_type": ["HUMAN PRESCRIPTION DRUG"],
                },
                "warnings": [long_warning],
            }
        ]
    }

    monkeypatch.setattr(
        "urllib.request.urlopen",
        lambda *args, **kwargs: _FakeHTTPResponse(payload),
    )

    skill = OpenFDASkill()
    results = skill.retrieve({"drug": ["clozapine"]}, max_results=6)

    warning = next(result for result in results if result.relationship == "has_warning")
    assert len(warning.evidence_text) <= 500
    assert not warning.evidence_text.endswith("disin")
    assert "disintegrate" not in warning.evidence_text
    assert warning.evidence_text.endswith("word")
