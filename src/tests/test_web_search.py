"""Web search: offered to Claude, run by Anthropic, never spoken raw.

Nothing here calls the API. What can break on our side is the registry's
bookkeeping and the two stream loop details a search introduces: a stage word
for the app, and the space between the text blocks it splits a reply into.
"""

from types import SimpleNamespace

import pytest

import brain
from tools import ToolError, ToolRegistry


def _start(block_type, name=None):
    return SimpleNamespace(type="content_block_start",
                           content_block=SimpleNamespace(type=block_type, name=name))


def _delta(text):
    return SimpleNamespace(type="content_block_delta",
                           delta=SimpleNamespace(type="text_delta", text=text))


# ── the gap between text blocks ──

def test_a_new_text_block_after_a_full_stop_gets_a_space():
    """Without it "Let me check." and "The Gators won." reach the splitter as
    one run, and its full stop needs whitespace after it to end a sentence."""
    assert brain._block_gap(_start("text"), "Let me check.") == " "


def test_no_space_at_the_start_of_a_reply():
    assert brain._block_gap(_start("text"), "") == ""


def test_no_second_space_when_one_is_already_there():
    assert brain._block_gap(_start("text"), "Let me check. ") == ""


def test_only_text_blocks_open_a_gap():
    assert brain._block_gap(_start("server_tool_use", "web_search"), "Hi.") == ""
    assert brain._block_gap(_start("web_search_tool_result"), "Hi.") == ""
    assert brain._block_gap(_delta("more"), "Hi.") == ""


# ── the stage word ──

def test_a_search_shows_a_stage_in_the_app():
    assert brain._stage_for(_start("server_tool_use", "web_search")) == "Searching the web"


def test_search_results_show_no_stage():
    assert brain._stage_for(_start("web_search_tool_result")) is None


# ── the registry ──

SCHEMA = {"type": "web_search_20250305", "name": "web_search", "max_uses": 2}


def test_a_server_tool_is_offered_but_not_executable():
    reg = ToolRegistry()
    reg.register_server_tool(SCHEMA, "Search the web")
    assert reg.api_schemas() == [SCHEMA]
    assert "web_search" not in reg
    with pytest.raises(ToolError):
        reg.get("web_search")


def test_a_server_tool_cannot_take_a_local_tools_name():
    reg = ToolRegistry()
    reg.register_server_tool(SCHEMA, "Search the web")
    with pytest.raises(ToolError):
        reg.register_server_tool(SCHEMA, "Search again")


def test_a_server_tool_needs_a_summary():
    """The summary is its line in the prompt's tool list. Without one Nova is
    told she does not have it."""
    with pytest.raises(ToolError):
        ToolRegistry().register_server_tool(SCHEMA, "  ")
