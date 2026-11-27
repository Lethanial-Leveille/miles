"""Preflight for the shared voice brain. Scheduling writes remain offline."""

import inspect
import json
import re
from dataclasses import dataclass, field
from threading import Event

from tools import Permission, ToolError, permits

GUARDED = {"task_add", "task_complete", "archive_task", "list_add", "list_check"}
CONFIRM = "confirm_pending_action"
PARALLEL_READS = {"get_weather", "tasks_today", "list_show"}
REFUSED = "Nothing was changed. I could not validate that request."
ONE_CHANGE = "Nothing was changed. Please ask for one change at a time."
_NO_TIME = re.compile(
    r"\b(?:start\w*|begin\w*|deadline|due|by|until|before|after|tomorrow|today|tonight|"
    r"mon(?:day)?|tue(?:sday)?|wed(?:nesday)?|thu(?:rsday)?|fri(?:day)?|sat(?:urday)?|sun(?:day)?|"
    r"january|february|march|april|may|june|july|august|september|october|november|december|morning|evening|afternoon|"
    r"week\w*|month\w*|year\w*|alternate|everyday|recurr\w*|repeat\w*|every|each|at|on|from|when|once|later|soon|"
    r"next|twice|in \d+)\b|\d{1,4}[-/:]\d", re.I)
_TASK_REQUEST = re.compile(
    r"^(?:please )?(?:archive|stop tracking|check off|cross off|complete|(?:add|create|put) .*(?:\btask\b|\blist\b|\bdaily\b|\bevery day\b))",
    re.I)
_ACTION_REQUEST = re.compile(r"^(?:please )?(?:add|create|archive|stop tracking|check off|cross off|complete|mark)\b", re.I)
_COMPOUND = re.compile(
    r"\b(?:and|then)\s+(?:add|create|archive|complete|mark|stop tracking|check off|put)\b", re.I)
_PREFIX = {
    "task_complete": r"(?:mark(?: as done)?|complete|finish|i (?:finished|completed|did|took|called|paid))",
    "archive_task": r"(?:archive|stop tracking|cancel)",
    "list_add": r"(?:add|put)",
    "list_check": r"(?:check off|cross off|i (?:got|bought))",
}


def _normalized(text):
    return " ".join(text.casefold().split()).strip(" .!?,")


def _intent_matches(name, args, utterance):
    text = _normalized(utterance).removeprefix("please ").removesuffix(" please")
    if name == "task_add":
        match = re.fullmatch(r"(?:add|create) (?:a task(?: to| for)? )?(.+)", text)
        if not match:
            return False
        title, count = re.subn(r"\s+(?:as a daily task|every day|each day|daily)$", "", match[1])
        return (bool(count) == (args.get("recurrence") == "daily")
                and not _NO_TIME.search(title) and title == _normalized(args["task"]))
    target = _normalized(args["task"] if "task" in args else args["item"])
    if not target:
        return False
    suffix = r"(?: (?:as )?done)?" if name == "task_complete" else ""
    if name.startswith("list_"):
        # Keep the existing grocery default. A nondefault list needs an explicit name.
        from lists_tasks import stored_list_name

        list_name = args.get("list_name")
        list_tail = ""
        if list_name:
            canonical = _normalized(stored_list_name(list_name))
            if not canonical:
                return False
            named = rf"(?: to| on| from| off) (?:my |the )?(?:{re.escape(canonical)} )?list"
            list_tail = rf"(?:{named})?" if canonical == "grocery" else rf"(?: to| on| from| off) (?:my |the )?{re.escape(canonical)} list"
        else:
            list_tail = r""
        suffix += list_tail
    return bool(re.fullmatch(rf"{_PREFIX[name]} (?:the |my )?{re.escape(target)}{suffix}", text))


def _valid_value(value, schema):
    kind = schema.get("type")
    if isinstance(kind, list):
        return any(_valid_value(value, {**schema, "type": item}) for item in kind)
    if "enum" in schema and not any(type(value) is type(item) and value == item for item in schema["enum"]):
        return False
    if kind == "object":
        if not isinstance(value, dict) or not set(schema.get("required", ())) <= value.keys():
            return False
        properties = schema.get("properties", {})
        return all((key in properties and _valid_value(item, properties[key]))
                   or (key not in properties and schema.get("additionalProperties") is not False)
                   for key, item in value.items())
    if kind == "array":
        return (isinstance(value, list) and len(value) >= schema.get("minItems", 0)
                and len(value) <= schema.get("maxItems", len(value))
                and all(_valid_value(item, schema.get("items", {})) for item in value))
    if kind == "string":
        return (isinstance(value, str) and len(value) >= schema.get("minLength", 0)
                and len(value) <= schema.get("maxLength", len(value))
                and ("pattern" not in schema or bool(re.search(schema["pattern"], value))))
    if kind == "boolean":
        return type(value) is bool
    if kind == "null":
        return value is None
    if kind in ("integer", "number"):
        expected = (int,) if kind == "integer" else (int, float)
        return (type(value) in expected and schema.get("minimum", value) <= value
                <= schema.get("maximum", value))
    return kind is None


def _valid_args(spec, args):
    schema = {**spec.input_schema, "additionalProperties": False}
    if not _valid_value(args, schema):
        return False
    try:
        inspect.signature(spec.func).bind(**args)
    except TypeError:
        return False
    return True


@dataclass
class TurnSafety:
    utterance: str
    cancelled: Event = field(default_factory=Event)
    guarded: bool = False
    used: bool = False
    prior_mutation: bool = False
    finished: dict = field(default_factory=dict)
    workers: list = field(default_factory=list)

    @property
    def expects_action(self):
        return bool(_ACTION_REQUEST.match(_normalized(self.utterance)))

    def preflight(self, blocks, registry, tier, can_confirm):
        """Decide the complete batch before any executor submission."""
        refusals = {}
        mutations = []
        seen = {}
        block_ids = set()
        for index, block in enumerate(blocks):
            try:
                spec = registry.get(block.name)
            except ToolError:
                refusals[index] = REFUSED
                mutations.append(index)
                continue
            if spec.permission is not Permission.READ:
                mutations.append(index)
            repeated_id = block.id in block_ids
            block_ids.add(block.id)
            if repeated_id:
                refusals[index] = REFUSED
            elif not permits(spec, tier):
                refusals[index] = "Nothing was changed by that action. It requires more clearance."
            elif not _valid_args(spec, block.input):
                refusals[index] = REFUSED
        candidates = [index for index in mutations if index not in refusals]
        covered = any(blocks[index].name in GUARDED or blocks[index].name == CONFIRM for index in mutations)
        self.guarded = self.guarded or covered or bool(_TASK_REQUEST.match(_normalized(self.utterance)))
        for index in candidates:
            block = blocks[index]
            key = (block.name, json.dumps(block.input, sort_keys=True))
            if block.id in self.finished or key in self.finished or key in seen:
                refusals[index] = "This action was already handled. Nothing was repeated."
            else:
                seen[key] = index
        distinct = [index for index in candidates if index not in refusals]
        # An invalid sibling must stop all writes, while vetted independent reads may proceed.
        blocked = any(index in refusals and index not in candidates for index in mutations)
        common = REFUSED if blocked else None
        if covered or self.guarded:
            if self.used or self.prior_mutation:
                common = "A new request is needed for another change."
            elif len(distinct) > 1 or _COMPOUND.search(self.utterance):
                common = ONE_CHANGE
            elif distinct:
                block = blocks[distinct[0]]
                if block.name not in GUARDED and block.name != CONFIRM:
                    common = REFUSED
                elif block.name in GUARDED and not _intent_matches(block.name, block.input, self.utterance):
                    common = REFUSED
                elif block.name == CONFIRM and not can_confirm(block.input["approved"]):
                    common = "Nothing was changed. There is no matching confirmation to apply."
            if mutations:
                self.used = True
        self.prior_mutation |= bool(mutations)
        if self.cancelled.is_set():
            common = "The request was cancelled."
        if common:
            for index in distinct:
                refusals[index] = common
        return refusals
