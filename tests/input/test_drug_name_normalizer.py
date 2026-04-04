from __future__ import annotations

from drugclaw.drug_alias_sources import InMemoryDrugAliasSource
from drugclaw.drug_name_normalizer import DrugNameNormalizer


def test_alias_source_maps_gleevec_to_imatinib() -> None:
    source = InMemoryDrugAliasSource.default()

    assert source.resolve_name("Gleevec") == "imatinib"


def test_alias_source_accepts_canonical_name() -> None:
    source = InMemoryDrugAliasSource.default()

    assert source.resolve_name("imatinib") == "imatinib"


def test_normalizer_resolves_alias_query_to_canonical_drug() -> None:
    normalizer = DrugNameNormalizer.default()

    result = normalizer.normalize_query("What does Gleevec target?")

    assert result["status"] == "resolved"
    assert result["canonical_drug_names"] == ["imatinib"]
    assert result["normalized_query"] == "What does imatinib target?"


def test_normalizer_handles_case_and_punctuation_for_aliases() -> None:
    normalizer = DrugNameNormalizer.default()

    result = normalizer.normalize_query("What prescribing and safety information is available for GLUCOPHAGE?")

    assert result["status"] == "resolved"
    assert result["canonical_drug_names"] == ["metformin"]
    assert result["normalized_query"].endswith("metformin?")


def test_normalizer_keeps_unknown_query_unresolved() -> None:
    normalizer = DrugNameNormalizer.default()

    result = normalizer.normalize_query("What does totally-unknown-drug target?")

    assert result["status"] == "unresolved"
    assert result["normalized_query"] == "What does totally-unknown-drug target?"


def test_normalizer_records_alias_and_canonical_mentions_for_same_drug_query() -> None:
    normalizer = DrugNameNormalizer.default()

    result = normalizer.normalize_query("What does Gleevec (imatinib) target?")

    assert result["status"] == "resolved"
    assert result["canonical_drug_names"] == ["imatinib"]
    assert result["rewrite_applied"] is True
    assert result["drug_mentions"] == [
        {
            "raw_text": "Gleevec",
            "mention_type": "alias",
            "canonical_drug_name": "imatinib",
            "resolution_stage": "name",
            "source": "alias_seed",
        },
        {
            "raw_text": "imatinib",
            "mention_type": "canonical_name",
            "canonical_drug_name": "imatinib",
            "resolution_stage": "name",
            "source": "alias_seed",
        },
    ]
