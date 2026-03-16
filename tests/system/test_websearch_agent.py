from drugclaw.agent_websearch import WebSearchAgent


def test_websearch_agent_does_not_build_fallback_adapters_when_skill_is_injected(
    monkeypatch,
) -> None:
    def _unexpected(*args, **kwargs):
        raise AssertionError("fallback adapter should not be initialized")

    monkeypatch.setattr("drugclaw.agent_websearch.PubMedAdapter", _unexpected)
    monkeypatch.setattr("drugclaw.agent_websearch.ClinicalTrialsAdapter", _unexpected)
    monkeypatch.setattr("drugclaw.agent_websearch.DuckDuckGoAdapter", _unexpected)
    monkeypatch.setattr("drugclaw.agent_websearch.BaiduAdapter", _unexpected)
    monkeypatch.setattr("drugclaw.agent_websearch.GoogleScholarAdapter", _unexpected)

    agent = WebSearchAgent(llm_client=object(), web_search_skill=object())

    assert agent._web_skill is not None
