"""Weather as a registered tool, returning structured data.

No network. Every test drives requests.get through a stub, because the point of
these is the shape of what comes back and the precipitation logic, neither of
which needs OpenWeatherMap to be up.
"""

import pytest

import actions
from tools import Permission


def _current(condition_id=800, description="clear sky", temp=90.7, feels=98.6,
             humidity=78, wind=5.3):
    return {
        "weather": [{"id": condition_id, "description": description}],
        "main": {"temp": temp, "feels_like": feels, "humidity": humidity},
        "wind": {"speed": wind},
    }


def _forecast(*condition_ids, start_hour=16):
    """Forecast blocks at three hour steps, ids given in order."""
    import datetime as dt
    base = dt.datetime(2026, 8, 11, start_hour, 0).timestamp()
    return {"list": [
        {"dt": base + i * 3 * 3600, "weather": [{"id": cid}]}
        for i, cid in enumerate(condition_ids)
    ]}


@pytest.fixture(autouse=True)
def clear_geocode_cache():
    actions._GEOCODE_CACHE.clear()
    yield
    actions._GEOCODE_CACHE.clear()


@pytest.fixture
def api(monkeypatch):
    """Stub requests.get, recording every URL hit so call counts are testable."""
    state = {"geo": [{"lat": 29.65, "lon": -82.32}], "current": _current(),
             "forecast": _forecast(800, 800, 800, 800), "urls": []}

    class Response:
        def __init__(self, payload):
            self._payload = payload

        def json(self):
            return self._payload

    def fake_get(url, params=None, **kwargs):
        state["urls"].append(url)
        if "geo/1.0/direct" in url:
            return Response(state["geo"])
        if "/data/2.5/weather" in url:
            return Response(state["current"])
        if "/data/2.5/forecast" in url:
            return Response(state["forecast"])
        raise AssertionError(f"unexpected URL {url}")

    monkeypatch.setattr(actions.requests, "get", fake_get)
    return state


# ── registration ──

def test_weather_is_registered_on_import():
    """prompts.build_enhanced_prompt reads the production registry, so weather
    has to land in it simply by actions being imported."""
    from tools import registry
    assert "get_weather" in registry
    spec = registry.get("get_weather")
    assert spec.permission is Permission.READ
    assert spec.returns_to_model is True


def test_schema_marks_location_optional():
    from tools import registry
    schema = registry.get("get_weather").input_schema
    assert schema["required"] == []
    assert "location" in schema["properties"]


def test_description_tells_the_model_when_to_call_it():
    """Under description is the common failure mode for tools. The trigger
    conditions matter more than the parameter list."""
    from tools import registry
    d = registry.get("get_weather").description.lower()
    assert "rain" in d and "jacket" in d


# ── return shape ──

def test_returns_a_dict_not_a_sentence(api):
    """The whole reason for step 3. A finished English paragraph gets read
    aloud as a paragraph no matter what the prompt says."""
    out = actions._fetch_weather("Gainesville")
    assert isinstance(out, dict)
    assert set(out) == {"location", "temp", "feels_like", "condition",
                        "humidity", "wind_mph", "precip"}


def test_temperatures_are_rounded(api):
    out = actions._fetch_weather("Gainesville")
    assert out["temp"] == 91
    assert out["feels_like"] == 99
    assert out["wind_mph"] == 5


def test_unknown_location_reports_rather_than_raising(api):
    api["geo"] = []
    assert "Could not find location" in actions._fetch_weather("Atlantis")["error"]


# ── precipitation outlook ──

def test_clear_now_and_clear_ahead_says_nothing(api):
    assert actions._fetch_weather("Gainesville")["precip"] is None


def test_clear_now_with_rain_coming_names_the_hour(api):
    api["forecast"] = _forecast(800, 800, 500, 500, start_hour=16)
    out = actions._fetch_weather("Gainesville")
    assert out["precip"] == "rain likely around 10 PM"


def test_raining_now_reports_when_it_eases(api):
    api["current"] = _current(condition_id=500, description="light rain")
    api["forecast"] = _forecast(500, 500, 800, 800, start_hour=16)
    out = actions._fetch_weather("Gainesville")
    assert out["precip"] == "easing off around 10 PM"


def test_raining_throughout_says_so(api):
    api["current"] = _current(condition_id=502, description="heavy rain")
    api["forecast"] = _forecast(500, 500, 500, 500)
    out = actions._fetch_weather("Gainesville")
    assert out["precip"] == "continuing through the next several hours"


@pytest.mark.parametrize("cid,is_precip", [
    (200, True),    # thunderstorm
    (300, True),    # drizzle
    (500, True),    # rain
    (600, True),    # snow
    (701, False),   # mist
    (800, False),   # clear
    (804, False),   # overcast
])
def test_precipitation_boundary(api, cid, is_precip):
    """741 fog is not rain. The boundary sits at 700 and getting it wrong makes
    Nova announce rain on a foggy morning."""
    api["forecast"] = _forecast(800, cid, 800, 800)
    out = actions._fetch_weather("Gainesville")
    assert (out["precip"] is not None) is is_precip


def test_forecast_failure_does_not_fail_the_lookup(api, monkeypatch):
    """Current conditions are still worth answering with."""
    real_get = actions.requests.get

    def flaky(url, params=None, **kwargs):
        if "/forecast" in url:
            raise ConnectionError("forecast down")
        return real_get(url, params=params, **kwargs)

    monkeypatch.setattr(actions.requests, "get", flaky)
    out = actions._fetch_weather("Gainesville")
    assert out["temp"] == 91
    assert out["precip"] is None


# ── geocode cache ──

def test_geocode_is_cached_across_calls(api):
    actions._fetch_weather("Gainesville")
    actions._fetch_weather("Gainesville")
    geo_calls = [u for u in api["urls"] if "geo/1.0/direct" in u]
    assert len(geo_calls) == 1


def test_different_locations_are_cached_separately(api):
    actions._fetch_weather("Gainesville")
    actions._fetch_weather("Orlando")
    geo_calls = [u for u in api["urls"] if "geo/1.0/direct" in u]
    assert len(geo_calls) == 2


# ── legacy prose path ──

def test_legacy_path_is_short(api):
    """Still prose, because the bracket path needs a string, but no longer the
    four fact paragraph that caused the verbosity complaint."""
    line = actions.get_weather("Gainesville")
    assert line == "Gainesville: 91 degrees, feels like 99, clear sky."
    assert "Humidity" not in line
    assert "Wind" not in line


def test_legacy_path_mentions_rain_when_there_is_rain(api):
    api["forecast"] = _forecast(800, 500, 800, 800, start_hour=16)
    assert "rain likely around 7 PM" in actions.get_weather("Gainesville")


def test_legacy_path_reports_errors_as_a_string(api):
    api["geo"] = []
    assert "Could not find location" in actions.get_weather("Atlantis")
