from __future__ import annotations

from skills.ddi.kegg_drug.kegg_drug_skill import KEGGDrugSkill


class _FakeHTTPResponse:
    def __init__(self, payload: str):
        self._payload = payload.encode()

    def read(self) -> bytes:
        return self._payload

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None


def test_kegg_drug_rest_find_resolves_partner_drug_names(monkeypatch) -> None:
    responses = {
        "https://rest.kegg.jp/find/drug/warfarin": "dr:D00682\tWarfarin\n",
        "https://rest.kegg.jp/ddi/dr:D00682": "dr:D00682\tdr:D00109\tCI\tEnzyme: CYP2C9\n",
        "https://rest.kegg.jp/get/dr:D00109": "ENTRY       D00109\nNAME        Aspirin; Acetylsalicylic acid\n",
    }

    def _fake_urlopen(url, timeout=20):
        return _FakeHTTPResponse(responses[url])

    monkeypatch.setattr("urllib.request.urlopen", _fake_urlopen)

    skill = KEGGDrugSkill()
    results = skill.retrieve({"drug": ["warfarin"]}, max_results=5)

    assert results
    assert results[0].target_entity == "Aspirin"
    assert results[0].metadata["target_id"] == "dr:D00109"
    assert results[0].metadata["target_name"] == "Aspirin"
