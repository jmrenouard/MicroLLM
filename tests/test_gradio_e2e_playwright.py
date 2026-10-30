"""Playwright End-to-End browser UI automation test for Gradio Decision Studio (Issue #147).

Verifies reactive workspace mode switching and multi-tab enterprise domain preset population
directly inside headless Chromium against the live Gradio Blocks application.
"""

from __future__ import annotations

import socket
from collections.abc import Generator

import pytest

playwright = pytest.importorskip("playwright")
from playwright.sync_api import sync_playwright

from microllm.gradio_app import create_gradio_demo


def _get_free_port() -> int:
    """Find an available TCP port for the ephemeral Gradio server."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


@pytest.fixture(scope="module")
def gradio_server() -> Generator[str, None, None]:
    """Launch Gradio Decision Studio in-process on an isolated ephemeral port."""
    port = _get_free_port()
    demo = create_gradio_demo()
    _app, _local_url, _ = demo.launch(
        server_name="127.0.0.1",
        server_port=port,
        prevent_thread_lock=True,
        quiet=True,
    )
    yield f"http://127.0.0.1:{port}"
    demo.close()


def test_gradio_workspace_mode_switcher_playwright(gradio_server: str) -> None:
    """Verify that switching workspace modes dynamically updates tab visibility and active tab selection."""
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context()
        page = context.new_page()

        page.goto(gradio_server, wait_until="networkidle")
        page.wait_for_selector("#workspace_mode_radio", timeout=12000)
        page.wait_for_selector("#main_tabs", timeout=12000)

        # 1. Verify Initial State: Simple Per-Type Tabs default
        tab1_btn = page.get_by_role("tab", name="✅ 1. Yes / No Question")
        assert tab1_btn.is_visible(), "Tab 1 button must be visible in default Simple mode"
        assert not page.locator("#cockpit_extra_panel").first.is_visible(), (
            "Cockpit extra drawer must be hidden in default Simple mode"
        )

        # 2. Switch to Advanced Studio & Custom JSON Mode
        page.locator("#workspace_mode_radio").get_by_text("Advanced Studio").click()
        page.wait_for_timeout(1000)

        # Tab 6 (Advanced Studio) must be active and visible
        tab6_btn = page.get_by_role("tab", name="⚙️ 6. Advanced Studio")
        assert tab6_btn.is_visible(), "Tab 6 button must be visible in Advanced Studio mode"
        assert page.locator("#tab_advanced").is_visible(), "Tab 6 content must be visible in Advanced Studio mode"
        assert not page.locator("#cockpit_extra_panel").first.is_visible(), (
            "Cockpit panel must remain hidden in Advanced Studio mode"
        )

        # 3. Switch to All-in-One Cockpit Mode (Full Studio)
        page.locator("#workspace_mode_radio").get_by_text("Cockpit Mode").click()
        page.wait_for_timeout(1000)

        # In Cockpit mode: cockpit_extra_panel must be visible and Tab 6 visible
        assert page.locator("#cockpit_extra_panel").first.is_visible(), (
            "Cockpit extra drawer must be visible in Cockpit mode"
        )
        assert tab6_btn.is_visible(), "Tab 6 must remain visible in Cockpit mode"

        # 4. Switch back to Simple Per-Type Tabs
        page.locator("#workspace_mode_radio").get_by_text("Simple Per-Type Tabs").click()
        page.wait_for_timeout(1000)

        assert tab1_btn.is_visible(), "Tab 1 button must be visible after returning to Simple mode"
        assert not page.locator("#cockpit_extra_panel").first.is_visible(), (
            "Cockpit extra drawer must be hidden after returning to Simple mode"
        )

        context.close()
        browser.close()


def test_gradio_preset_loading_across_tabs_playwright(gradio_server: str) -> None:
    """Verify that selecting an enterprise preset populates form fields across Tabs 1, 2, and 6."""
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context()
        page = context.new_page()

        page.goto(gradio_server, wait_until="networkidle")
        page.wait_for_selector("#preset_dropdown", timeout=12000)

        # 1. Open the preset dropdown and select Cold-Chain scenario
        page.locator("#preset_dropdown input").click()
        page.wait_for_timeout(500)
        cold_chain_item = page.locator("ul.options li, .options .item, li[role=option]").filter(has_text="Cold-Chain")
        assert cold_chain_item.count() >= 1, "Cold-Chain Logistics preset option must exist in dropdown"
        cold_chain_item.first.click()
        page.wait_for_timeout(1200)

        # 2. Verify Tab 1 (Yes/No Question) was populated with cold-chain scenario data
        yn_textarea_values = [t.input_value() for t in page.locator("#tab_yn textarea").all()]
        has_cold_chain_yn = any("RF-8821" in val or "vaccine" in val.lower() for val in yn_textarea_values)
        assert has_cold_chain_yn, f"Tab 1 must reflect Cold-Chain scenario. Got: {yn_textarea_values}"

        # 3. Switch to Tab 2 (Single Choice) and verify form fields populated
        page.get_by_role("tab", name="🎯 2. Single Choice (Pick 1)").click()
        page.wait_for_timeout(500)
        ch_textarea_values = [t.input_value() for t in page.locator("#tab_choice textarea").all()]
        has_cold_chain_choice = any("RF-8821" in val or "quarantine" in val.lower() for val in ch_textarea_values)
        assert has_cold_chain_choice, f"Tab 2 must reflect Cold-Chain scenario. Got: {ch_textarea_values}"

        # 4. Switch to Advanced Studio Mode and verify Tab 6 state_input populated
        page.locator("#workspace_mode_radio").get_by_text("Advanced Studio").click()
        page.wait_for_timeout(1000)
        tab6_textarea_values = [t.input_value() for t in page.locator("#tab_advanced textarea").all()]
        has_cold_chain_tab6 = any("RF-8821" in val for val in tab6_textarea_values)
        assert has_cold_chain_tab6, f"Tab 6 must reflect Cold-Chain scenario. Got: {tab6_textarea_values}"

        context.close()
        browser.close()
