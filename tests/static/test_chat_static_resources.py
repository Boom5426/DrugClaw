from __future__ import annotations

from pathlib import Path


def test_chat_page_includes_resource_registry_panel_mount_points() -> None:
    html = Path("drugclaw/static/chat.html").read_text(encoding="utf-8")

    assert 'id="resources-panel"' in html
    assert 'id="resource-summary"' in html
    assert 'id="resource-list"' in html


def test_chat_script_fetches_resources_and_renders_package_health_fields() -> None:
    script = Path("drugclaw/static/chat.js").read_text(encoding="utf-8")

    assert 'fetch("/resources")' in script
    assert "package_status_counts" in script
    assert "gateway_ready_resources" in script
    assert "resource-list" in script

