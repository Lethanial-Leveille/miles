"""Weather as a registered tool, returning structured data.

No network. Every test drives requests.get through a stub, because the point of
these is the shape of what comes back and the precipitation logic, neither of
which needs Apple to be up. Shapes copied from real WeatherKit and Maps
responses on Nov 21 2026.
"""
from datetime import datetime, timedelta, timezone

import pytest

import actions
import apple_keys
from tools import Permission


def _current(code="Clear", temp_c=32.6, feels_c=37.0, humidity=0.78, wind_kmh=8.5,
             intensity=0.0):
    return {"conditionCode": code, "temperature": temp_c, "temperatureApparent": feels_c,
            "humidity": humidity, "windSpeed": wind_kmh, "precipitationIntensity": intensity}


def _hours(*wet, start=None, kind="rain"):
    """Hourly forecast beginning with the current hour, then one per flag.

    The first entry is the hour already under way, which the outlook skips."""
    start = start or datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    out = [{"forecastStart": start.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "precipitationType": "clear", "precipitationChance": 0.0}]
    for i, w in enumerate(wet, 1):
        t = start + timedelta(hours=i)
        out.append({"forecastStart": t.strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "precipitationType": kind if w else "clear",
                    "precipitationChance": w if isinstance(w, float) else (0.8 if w else 0.05)})
    return out


def _hour_label(i):
    start = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    return (start + timedelta(hours=i)).astimezone().strftime("%-I %p")


@pytest.fixture(autouse=True)
def clear_caches(monkeypatch):
    actions._GEOCODE_CACHE.clear()
    monkeypatch.setattr(apple_keys, "configured", lambda: True)
    monkeypatch.setattr(apple_keys, "weatherkit_token", lambda: "wk")
    monkeypatch.setattr(apple_keys, "maps_access_token", lambda: "maps")
    yield
    actions._GEOCODE_CACHE.clear()


@pytest.fixture
def api(monkeypatch):
    """Stub requests.get, recording every URL hit so call counts are testable."""
    state = {"geo": [{"coordinate": {"latitude": 29.65, "longitude": -82.32},
                      "formattedAddressLines": ["Gainesville, FL", "United States"]}],
             "current": _current(), "hours": _hours(False, False, False, False),
             "urls": [], "headers": []}

    class Response:
        def __init__(self, payload):
            self._payload = payload

        def json(self):
            return self._payload

    def fake_get(url, params=None, headers=None, **kwargs):
        state["urls"].append(url)
        state["headers"].append(headers)
        if "maps-api.apple.com/v1/geocode" in url:
            return Response({"results": state["geo"]})
        if "weatherkit.apple.com" in url:
            return Response({"currentWeather": state["current"],
                             "forecastHourly": {"hours": state["hours"]}})
        raise AssertionError(f"unexpected URL {url}")

    monkeypatch.setattr(actions.requests, "get", fake_get)
    return state


# ── registration ──

def test_weather_is_registered_on_import():
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
    from tools import registry
    d = registry.get("get_weather").description.lower()
    assert "rain" in d and "jacket" in d


# ── return shape ──

def test_returns_a_dict_not_a_sentence(api):
    out = actions.fetch_weather("Gainesville")
    assert isinstance(out, dict)
    assert set(out) == {"location", "temp", "feels_like", "condition",
                        "humidity", "wind_mph", "precip"}


def test_units_are_converted_and_rounded(api):
    """WeatherKit answers in Celsius, km/h and a 0 to 1 humidity."""
    out = actions.fetch_weather("Gainesville")
    assert out["temp"] == 91
    assert out["feels_like"] == 99
    assert out["wind_mph"] == 5
    assert out["humidity"] == 78


def test_condition_code_reads_as_words(api):
    api["current"] = _current(code="MostlyCloudy")
    assert actions.fetch_weather("Gainesville")["condition"] == "mostly cloudy"


def test_location_is_the_place_apple_found(api):
    """So a wrong match shows up in her answer instead of passing silently."""
    assert actions.fetch_weather("Gainesville")["location"] == "Gainesville, FL"


def test_both_services_get_their_own_token(api):
    actions.fetch_weather("Gainesville")
    assert {"Authorization": "Bearer maps"} in api["headers"]
    assert {"Authorization": "Bearer wk"} in api["headers"]


def test_unknown_location_reports_rather_than_raising(api):
    api["geo"] = []
    assert "Could not find location" in actions.fetch_weather("Atlantis")["error"]


def test_missing_key_reports_rather_than_raising(api, monkeypatch):
    monkeypatch.setattr(apple_keys, "configured", lambda: False)
    assert "Apple key" in actions.fetch_weather()["error"]
    assert api["urls"] == []


# ── precipitation outlook ──

def test_clear_now_and_clear_ahead_says_nothing(api):
    assert actions.fetch_weather("Gainesville")["precip"] is None


def test_clear_now_with_rain_coming_names_the_hour(api):
    api["hours"] = _hours(False, False, True, True)
    assert actions.fetch_weather("Gainesville")["precip"] == f"rain likely around {_hour_label(3)}"


def test_snow_is_named_as_snow(api):
    api["hours"] = _hours(False, True, kind="snow")
    assert actions.fetch_weather("Gainesville")["precip"] == f"snow likely around {_hour_label(2)}"


def test_a_low_chance_is_not_announced(api):
    """A 30% chance most summer afternoons carry is not worth saying."""
    api["hours"] = _hours(0.3, 0.4, 0.45)
    assert actions.fetch_weather("Gainesville")["precip"] is None


def test_raining_now_reports_when_it_eases(api):
    api["current"] = _current(code="Rain", intensity=1.2)
    api["hours"] = _hours(True, True, False, False)
    assert actions.fetch_weather("Gainesville")["precip"] == f"easing off around {_hour_label(3)}"


def test_raining_throughout_says_so(api):
    api["current"] = _current(code="HeavyRain", intensity=4.0)
    api["hours"] = _hours(True, True, True, True)
    assert actions.fetch_weather("Gainesville")["precip"] == "continuing through the next several hours"


def test_the_hour_already_under_way_is_skipped(api):
    """Its rain is the current conditions, not something coming."""
    api["hours"] = _hours(False, False)
    api["hours"][0].update(precipitationType="rain", precipitationChance=0.9)
    assert actions.fetch_weather("Gainesville")["precip"] is None


@pytest.mark.parametrize("code,wet", [
    ("Rain", True), ("Drizzle", True), ("Thunderstorms", True), ("Snow", True),
    ("Foggy", False), ("Haze", False), ("Clear", False), ("Cloudy", False),
])
def test_fog_is_not_rain(api, code, wet):
    """Fog must not read as rain, or Nova announces rain on a foggy morning."""
    api["current"] = _current(code=code)
    api["hours"] = _hours(False, False)
    out = actions.fetch_weather("Gainesville")
    assert (out["precip"] is not None) is wet


def test_forecast_problem_does_not_fail_the_lookup(api):
    api["hours"] = [{"broken": True}]
    out = actions.fetch_weather("Gainesville")
    assert out["temp"] == 91
    assert out["precip"] is None


# ── geocode cache ──

def test_geocode_is_cached_across_calls(api):
    actions.fetch_weather("Gainesville")
    actions.fetch_weather("Gainesville")
    assert len([u for u in api["urls"] if "geocode" in u]) == 1


def test_different_locations_are_cached_separately(api):
    actions.fetch_weather("Gainesville")
    actions.fetch_weather("Orlando")
    assert len([u for u in api["urls"] if "geocode" in u]) == 2
