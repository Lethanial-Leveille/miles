import netcheck
import phrasebank

# A real /proc/net/route from this Pi, with the docker0 and bridge lines that
# make link state alone unreliable. wlan0 is the only default route.
ROUTE_WITH_DEFAULT = """\
Iface\tDestination\tGateway \tFlags\tRefCnt\tUse\tMetric\tMask\t\tMTU\tWindow\tIRTT
wlan0\t00000000\t0100A8C0\t0003\t0\t0\t600\t00000000\t0\t0\t0
docker0\t000011AC\t00000000\t0001\t0\t0\t0\t0000FFFF\t0\t0\t0
wlan0\t0000A8C0\t00000000\t0001\t0\t0\t600\t00FFFFFF\t0\t0\t0
"""

# Same machine with the uplink gone. The docker bridge survives, which is the
# whole reason the default route is what gets checked rather than link state.
ROUTE_NO_DEFAULT = """\
Iface\tDestination\tGateway \tFlags\tRefCnt\tUse\tMetric\tMask\t\tMTU\tWindow\tIRTT
docker0\t000011AC\t00000000\t0001\t0\t0\t0\t0000FFFF\t0\t0\t0
"""


def _route_file(tmp_path, contents):
    path = tmp_path / "route"
    path.write_text(contents)
    return str(path)


def test_finds_the_default_route_interface(tmp_path):
    assert netcheck.default_interface(
        _route_file(tmp_path, ROUTE_WITH_DEFAULT)) == "wlan0"


def test_docker_bridge_alone_is_not_a_default_route(tmp_path):
    assert netcheck.default_interface(
        _route_file(tmp_path, ROUTE_NO_DEFAULT)) is None


def test_missing_route_table_is_not_an_error(tmp_path):
    assert netcheck.default_interface(str(tmp_path / "absent")) is None


def test_link_up_reads_operstate(tmp_path):
    (tmp_path / "wlan0").mkdir()
    (tmp_path / "wlan0" / "operstate").write_text("up\n")
    assert netcheck.link_up("wlan0", sysfs=str(tmp_path))


def test_link_down_reads_operstate(tmp_path):
    (tmp_path / "wlan0").mkdir()
    (tmp_path / "wlan0" / "operstate").write_text("down\n")
    assert not netcheck.link_up("wlan0", sysfs=str(tmp_path))


def test_absent_interface_is_not_up(tmp_path):
    assert not netcheck.link_up("wlan9", sysfs=str(tmp_path))


def _patch(monkeypatch, iface, up, internet, dns):
    monkeypatch.setattr(netcheck, "default_interface", lambda *a, **k: iface)
    monkeypatch.setattr(netcheck, "link_up", lambda *a, **k: up)
    monkeypatch.setattr(netcheck, "reachable", lambda *a, **k: internet)
    monkeypatch.setattr(netcheck, "resolves", lambda *a, **k: dns)


def test_no_default_route_is_no_wifi(monkeypatch):
    _patch(monkeypatch, None, False, False, False)
    assert netcheck.diagnose() == 'no_wifi'


def test_route_present_but_link_down_is_no_wifi(monkeypatch):
    _patch(monkeypatch, "wlan0", False, False, False)
    assert netcheck.diagnose() == 'no_wifi'


def test_link_up_but_nothing_routes_out_is_no_internet(monkeypatch):
    _patch(monkeypatch, "wlan0", True, False, False)
    assert netcheck.diagnose() == 'no_internet'


def test_internet_reachable_but_name_fails_is_dns(monkeypatch):
    _patch(monkeypatch, "wlan0", True, True, False)
    assert netcheck.diagnose() == 'no_dns'


def test_everything_local_healthy_blames_the_api(monkeypatch):
    _patch(monkeypatch, "wlan0", True, True, True)
    assert netcheck.diagnose() == 'api_down'


def test_every_diagnosis_has_phrases_to_say():
    """The whole point is that she can speak the cause. A key diagnose can
    return with no entry in PHRASES would fall through to tts.speak, which
    needs the network that just failed."""
    for cause in ('no_wifi', 'no_internet', 'no_dns', 'api_down'):
        assert phrasebank.PHRASES.get(cause), cause


def test_timer_alert_uses_an_attributive_singular():
    """"Your five minutes timer is up" is wrong: before the noun the unit
    modifies it and must be singular. The quantity form stays plural."""
    import actions
    assert actions._attributive("minutes") == "minute"
    assert actions._attributive("minute") == "minute"
    assert actions._plural(5, "minutes") == "minutes"
    assert actions._plural(1, "minutes") == "minute"


def test_timer_alert_spells_the_number():
    import actions
    assert actions._spoken_amount(5) == "five"
    assert actions._spoken_amount(45) == "forty five"
    # The speller widened past sixty when local weather needed to say
    # temperatures aloud, so a ninety minute timer alert spells it now rather
    # than reading the digits. Strictly better and worth pinning as intended.
    assert actions._spoken_amount(90) == "ninety"
    assert actions._spoken_amount(100) == "one hundred"
    # Still digits rather than a crash once past what the speller covers.
    assert actions._spoken_amount(500) == "500"
