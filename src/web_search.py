"""Web search, offered to Claude and run by Anthropic, not by this process.

Everything else Nova can do is a function here that the executor calls. This
one is not: the search happens on Anthropic's side, inside the same streaming
call, and its results come back as content blocks the stream loop never speaks.
So there is no function, no permission tier and no tool_call_log row. The only
thing this module does is put the tool in the request and in the prompt's tool
list, and only when WEB_SEARCH is on.

Every tier gets it, a guest included. What it reaches is public, and the cost
is capped per turn by max_uses.
"""

from config import (WEB_SEARCH, WEB_SEARCH_LOCATION, WEB_SEARCH_MAX_USES,
                    WEB_SEARCH_TOOL_TYPE)
from tools import registry

SCHEMA = {
    "type": WEB_SEARCH_TOOL_TYPE,
    "name": "web_search",
    "max_uses": WEB_SEARCH_MAX_USES,
    "user_location": WEB_SEARCH_LOCATION,
}

SUMMARY = ("Search the web for anything current or anything you do not know: "
           "scores, news, prices, hours, releases, what happened today")

if WEB_SEARCH:
    registry.register_server_tool(SCHEMA, SUMMARY)
