"""Tool registry for native Anthropic tool use.

Replaces the bracket action tag system, where adding one capability meant
consistent edits in five places (prompt prose, extract_actions,
_parse_action_tag, execute_actions, and the needs_data whitelist) with nothing
enforcing that they agreed. Four more tools would have been sixteen chances to
miss one, and the failure mode is a capability the prompt does not know exists.

Everything the rest of the system needs about a tool comes from here: the
schemas sent to the API, the capability block in the system prompt, the
permission tier, whether a call needs a second round trip, and the function to
run. One registration, one source of truth.

Nothing in this module talks to the API. It is a registry and nothing else, so
it can be tested without a network call or an event loop.
"""

import re
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable


class Permission(Enum):
    """What a tool is allowed to touch.

    Defined now, enforced later. Recording the tier at registration is cheap;
    retrofitting it across a dozen tools after the fact is not, and a tool
    added without one would be the tool that needed it most.

    READ            reads state, changes nothing (weather, system state, Hevy)
    WRITE           changes state on this Pi (timers, reminders)
    EXTERNAL_WRITE  changes state on someone else's system (calendar writes)
    CONTROL         changes Nova's own conversational state (dismiss)
    """
    READ = "read"
    WRITE = "write"
    EXTERNAL_WRITE = "external_write"
    CONTROL = "control"


# Lowercase and underscores only. The API permits hyphens and uppercase; this
# is deliberately narrower, partly to match the no hyphens rule and partly
# because one tool named get_weather and another named get-weather would be a
# miserable afternoon.
_NAME_RULE = re.compile(r"^[a-z][a-z0-9_]*$")


class ToolError(Exception):
    """Raised for registration mistakes and unknown tool lookups."""


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    input_schema: dict
    permission: Permission
    # True when the model has to speak about what came back, which costs a
    # second Claude call. False for fire and forget work, where the text the
    # model already produced alongside the call is the final answer. This is
    # the field that replaces the hardcoded needs_data whitelist in brain.py.
    returns_to_model: bool
    func: Callable[..., Any]

    @property
    def summary(self) -> str:
        """First sentence of the description, for the capability block.

        Deliberately derived rather than stored as its own field. A separate
        summary is one more thing that can drift from the description, and
        this way a vague first sentence shows up in the prompt where it will
        be noticed."""
        head = self.description.strip().split(". ")[0]
        return head.rstrip(".")

    def api_schema(self) -> dict:
        """Exactly what the API accepts, and nothing else.

        permission, returns_to_model, and func are ours. Sending them would be
        rejected, and more to the point they are internal routing decisions
        that the model has no business seeing."""
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema,
        }


class ToolRegistry:
    """Holds registered tools and derives everything downstream from them.

    A class rather than module globals so tests can build a throwaway registry
    instead of mutating the one production uses.
    """

    def __init__(self) -> None:
        self._tools: dict[str, ToolSpec] = {}

    # ── registration ──

    def register(
        self,
        *,
        name: str,
        description: str,
        input_schema: dict,
        permission: Permission,
        returns_to_model: bool,
    ) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        """Decorator. Registers a function as a tool and returns it unchanged.

        Returning the original function rather than a wrapper matters: every
        tool stays an ordinary function that can be called and tested directly,
        and registration can never alter behavior at runtime.
        """
        self._validate(name, description, input_schema, permission, returns_to_model)

        def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
            self._tools[name] = ToolSpec(
                name=name,
                description=description,
                input_schema=input_schema,
                permission=permission,
                returns_to_model=returns_to_model,
                func=func,
            )
            return func

        return decorator

    def _validate(self, name, description, input_schema, permission, returns_to_model) -> None:
        """Fail at import time rather than mid conversation.

        Every check here is something that would otherwise surface as a live
        turn behaving strangely with no error anywhere."""
        if not isinstance(name, str) or not _NAME_RULE.match(name):
            raise ToolError(
                f"tool name {name!r} must be lowercase letters, digits and "
                "underscores, starting with a letter"
            )
        if name in self._tools:
            raise ToolError(f"tool {name!r} is already registered")
        if not isinstance(description, str) or not description.strip():
            raise ToolError(f"tool {name!r} needs a non empty description")
        if not isinstance(permission, Permission):
            raise ToolError(f"tool {name!r} needs a Permission, got {permission!r}")
        if not isinstance(returns_to_model, bool):
            raise ToolError(f"tool {name!r} needs returns_to_model as a bool")

        if not isinstance(input_schema, dict):
            raise ToolError(f"tool {name!r} input_schema must be a dict")
        if input_schema.get("type") != "object":
            raise ToolError(f"tool {name!r} input_schema needs type 'object'")
        properties = input_schema.get("properties")
        if not isinstance(properties, dict):
            raise ToolError(f"tool {name!r} input_schema needs a properties dict")

        # A name in `required` that is not in `properties` is a typo the API
        # will not catch. It surfaces as the model omitting a parameter it was
        # told was mandatory, with no error raised anywhere.
        for field in input_schema.get("required", []):
            if field not in properties:
                raise ToolError(
                    f"tool {name!r} marks {field!r} required but does not define it"
                )

    # ── lookup ──

    def get(self, name: str) -> ToolSpec:
        try:
            return self._tools[name]
        except KeyError:
            raise ToolError(f"unknown tool {name!r}") from None

    def names(self) -> list[str]:
        return sorted(self._tools)

    def __contains__(self, name: object) -> bool:
        return name in self._tools

    def __len__(self) -> int:
        return len(self._tools)

    # ── derived output ──

    def api_schemas(self) -> list[dict]:
        """Tool definitions for the API, sorted by name.

        The sort is load bearing, not tidiness. Tools render ahead of the
        system prompt in the cached prefix, so if this list ever reordered,
        every byte after it would move and every cache hit would silently
        disappear. Sorting makes the output depend only on which tools exist,
        never on the order they happened to be imported in."""
        return [self._tools[name].api_schema() for name in self.names()]

    def capability_prose(self) -> str:
        """The capability block for the system prompt, generated from the
        registry so the prompt can never claim a capability the code lacks.

        Kept to one line per tool. The full descriptions already reach the
        model as tool schemas, so repeating them here would pay for the same
        tokens twice. What this block adds is the boundary: the statement that
        the list is complete, which is what Nova needs to tell a missing
        capability from one she has forgotten how to reach."""
        if not self._tools:
            return ""

        lines = [
            "YOUR TOOLS:",
            "These are the only tools you have. Their parameters are supplied "
            "separately alongside each one.",
            "",
        ]
        lines += [f"- {self._tools[name].summary}" for name in self.names()]
        lines += [
            "",
            "If something you are asked for is not on this list, you do not have "
            "it, and you say so rather than inventing the result.",
        ]
        return "\n".join(lines)

    def call(self, name: str, arguments: dict) -> Any:
        """Run a registered tool.

        Argument mismatches are left to raise TypeError from the call itself.
        That is a programming error between the schema and the signature, it
        should be loud, and wrapping it would only hide which parameter was
        wrong."""
        return self.get(name).func(**arguments)


# Production registry. Tools register against this at import; tests build their
# own so they cannot contaminate it.
registry = ToolRegistry()
tool = registry.register
