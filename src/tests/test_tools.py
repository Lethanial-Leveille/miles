import pytest

from tools import Permission, ToolError, ToolRegistry


def _schema(properties=None, required=None):
    """Minimal valid input schema, with hooks for the invalid variants."""
    return {
        "type": "object",
        "properties": properties if properties is not None else {"city": {"type": "string"}},
        "required": required if required is not None else [],
    }


@pytest.fixture
def reg():
    """A throwaway registry. Tests never touch the production singleton, so no
    test can leak a tool into another test or into a live turn."""
    return ToolRegistry()


def _add(reg, name="get_weather", description="Current outdoor conditions. Extra detail here.",
         schema=None, permission=Permission.READ, returns_to_model=True):
    @reg.register(
        name=name,
        description=description,
        input_schema=schema if schema is not None else _schema(),
        permission=permission,
        returns_to_model=returns_to_model,
    )
    def handler(**kwargs):
        return {"called": name, "args": kwargs}

    return handler


# ── registration ──

def test_registers_and_retrieves(reg):
    _add(reg)
    assert "get_weather" in reg
    assert len(reg) == 1
    spec = reg.get("get_weather")
    assert spec.permission is Permission.READ
    assert spec.returns_to_model is True


def test_decorator_returns_the_function_unchanged(reg):
    """Registration is a side effect, never a wrapper. A tool has to stay an
    ordinary function that can be called and tested directly."""
    def plain(city):
        return f"weather in {city}"

    decorated = reg.register(
        name="get_weather",
        description="Current conditions.",
        input_schema=_schema(),
        permission=Permission.READ,
        returns_to_model=True,
    )(plain)

    assert decorated is plain
    assert decorated("Gainesville") == "weather in Gainesville"


def test_duplicate_name_is_rejected(reg):
    _add(reg)
    with pytest.raises(ToolError, match="already registered"):
        _add(reg)


@pytest.mark.parametrize("bad", ["GetWeather", "get-weather", "2weather", "", "get weather"])
def test_invalid_names_are_rejected(reg, bad):
    with pytest.raises(ToolError, match="tool name"):
        _add(reg, name=bad)


def test_blank_description_is_rejected(reg):
    with pytest.raises(ToolError, match="non empty description"):
        _add(reg, description="   ")


def test_permission_must_be_the_enum(reg):
    with pytest.raises(ToolError, match="needs a Permission"):
        _add(reg, permission="read")


def test_returns_to_model_must_be_bool(reg):
    with pytest.raises(ToolError, match="returns_to_model"):
        _add(reg, returns_to_model="yes")


@pytest.mark.parametrize("schema", [
    {"type": "string", "properties": {}},          # wrong type
    {"type": "object"},                            # no properties
    {"type": "object", "properties": ["city"]},    # properties not a dict
])
def test_malformed_schema_is_rejected(reg, schema):
    with pytest.raises(ToolError):
        _add(reg, schema=schema)


def test_required_naming_an_undefined_property_is_rejected(reg):
    """The check that earns its keep. The API will not catch this; it surfaces
    as the model omitting a parameter it was told was mandatory, with no error
    raised anywhere."""
    schema = _schema(properties={"location": {"type": "string"}}, required=["locaton"])
    with pytest.raises(ToolError, match="marks 'locaton' required"):
        _add(reg, schema=schema)


def test_failed_registration_leaves_the_registry_clean(reg):
    with pytest.raises(ToolError):
        _add(reg, name="BadName")
    assert len(reg) == 0


# ── derived output ──

def test_api_schemas_are_sorted_by_name(reg):
    """Sorting is load bearing. Tools render ahead of the system prompt in the
    cached prefix, so registration order leaking into this list would move
    every byte after it and silently drop every cache hit."""
    _add(reg, name="set_timer", description="Start a countdown.")
    _add(reg, name="get_weather", description="Current conditions.")
    _add(reg, name="dismiss", description="End the conversation.")

    assert [s["name"] for s in reg.api_schemas()] == ["dismiss", "get_weather", "set_timer"]


def test_api_schema_carries_only_api_fields(reg):
    """permission, returns_to_model and func are internal routing. The API
    rejects unknown keys, and the model has no business seeing them."""
    _add(reg)
    schema = reg.api_schemas()[0]
    assert set(schema) == {"name", "description", "input_schema"}


def test_summary_is_the_first_sentence(reg):
    _add(reg, description="Current outdoor conditions. Mentions rain when it is coming.")
    assert reg.get("get_weather").summary == "Current outdoor conditions"


def test_summary_handles_a_single_sentence_description(reg):
    _add(reg, description="End the conversation.")
    assert reg.get("get_weather").summary == "End the conversation"


def test_capability_prose_lists_every_tool(reg):
    _add(reg, name="get_weather", description="Current outdoor conditions. More.")
    _add(reg, name="set_timer", description="Start a countdown. More.")

    prose = reg.capability_prose()
    assert "Current outdoor conditions" in prose
    assert "Start a countdown" in prose
    # The boundary statement is the reason this block exists at all.
    assert "only tools you have" in prose


def test_capability_prose_is_empty_when_nothing_is_registered(reg):
    """prompts.py concatenates this directly, so an empty registry has to
    produce nothing rather than a dangling header."""
    assert reg.capability_prose() == ""


def test_capability_prose_order_matches_schema_order(reg):
    _add(reg, name="set_timer", description="Start a countdown.")
    _add(reg, name="get_weather", description="Current conditions.")

    prose = reg.capability_prose()
    assert prose.index("Current conditions") < prose.index("Start a countdown")


# ── dispatch ──

def test_call_dispatches_to_the_function(reg):
    _add(reg)
    assert reg.call("get_weather", {"city": "Gainesville"}) == {
        "called": "get_weather",
        "args": {"city": "Gainesville"},
    }


def test_call_on_unknown_tool_raises(reg):
    with pytest.raises(ToolError, match="unknown tool"):
        reg.call("get_stock_price", {})


def test_get_on_unknown_tool_raises(reg):
    with pytest.raises(ToolError, match="unknown tool"):
        reg.get("nope")


def test_names_are_sorted(reg):
    _add(reg, name="set_timer", description="Start a countdown.")
    _add(reg, name="dismiss", description="End the conversation.")
    assert reg.names() == ["dismiss", "set_timer"]


# ── the production singleton ──

def test_production_registry_is_importable_and_separate():
    """tools.registry is what production uses. Nothing registers against it
    yet; weather is step 3. This exists so that stays true by accident rather
    than by luck."""
    import tools
    assert isinstance(tools.registry, ToolRegistry)
    # `is` cannot work here: registry.register is a bound method, so every
    # attribute access builds a new object. What matters is what it is bound to.
    assert tools.tool.__self__ is tools.registry
