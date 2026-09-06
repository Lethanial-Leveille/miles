import config

# The real /proc/asound/cards from this Pi, which is what makes the bug this
# file guards against concrete: the microphone is card 3 and card 0 is the USB
# speaker adapter. MIC_MIXER_CARD was hardcoded to "0", so every gain check
# read the speakers and warned about drift at every service start.
CARDS_REAL = """\
 0 [Audio          ]: USB-Audio - AB13X USB Audio
                      Generic AB13X USB Audio at usb-xhci-hcd.1-2, full speed
 1 [vc4hdmi0       ]: vc4-hdmi - vc4-hdmi-0
                      vc4-hdmi-0
 2 [vc4hdmi1       ]: vc4-hdmi - vc4-hdmi-1
                      vc4-hdmi-1
 3 [Mini           ]: USB-Audio - Razer Seiren V3 Mini
                      Razer Inc. Razer Seiren V3 Mini at usb-xhci-hcd.0-1, full speed
"""

# Same machine after a reboot reordered the USB enumeration. The point of
# resolving by name is that this must give a different answer than the file
# above, without anything else changing.
CARDS_REORDERED = """\
 0 [Mini           ]: USB-Audio - Razer Seiren V3 Mini
                      Razer Inc. Razer Seiren V3 Mini at usb-xhci-hcd.0-1, full speed
 1 [Audio          ]: USB-Audio - AB13X USB Audio
                      Generic AB13X USB Audio at usb-xhci-hcd.1-2, full speed
"""

# The mic unplugged. Nothing here matches the hint.
CARDS_NO_MIC = """\
 0 [Audio          ]: USB-Audio - AB13X USB Audio
                      Generic AB13X USB Audio at usb-xhci-hcd.1-2, full speed
 1 [vc4hdmi0       ]: vc4-hdmi - vc4-hdmi-0
                      vc4-hdmi-0
"""


def _cards_file(tmp_path, contents):
    path = tmp_path / "cards"
    path.write_text(contents)
    return str(path)


def test_finds_the_mic_card_by_name(tmp_path):
    assert config._resolve_mixer_card(
        "Seiren", _cards_file(tmp_path, CARDS_REAL)) == "3"


def test_does_not_return_the_speaker_card(tmp_path):
    """The regression this file exists for.

    Card 0 is the AB13X speaker adapter and its capture control runs 0 to 255
    against the microphone's 0 to 31, so the two do not even share a scale.
    Reading 23 off the wrong one is not a smaller error than reading nothing,
    it is a different measurement presented as the right one."""
    assert config._resolve_mixer_card(
        "Seiren", _cards_file(tmp_path, CARDS_REAL)) != "0"


def test_follows_the_mic_when_card_numbers_shift(tmp_path):
    assert config._resolve_mixer_card(
        "Seiren", _cards_file(tmp_path, CARDS_REORDERED)) == "0"


def test_matches_a_hint_that_only_appears_on_the_detail_line(tmp_path):
    """The card id is "Mini" and the vendor only shows up in the description,
    so a matcher that read the first line alone would miss most hints."""
    assert config._resolve_mixer_card(
        "Razer Inc.", _cards_file(tmp_path, CARDS_REAL)) == "3"


def test_returns_none_when_the_mic_is_absent(tmp_path):
    """None rather than a fallback card number.

    A fallback does not fail, it silently measures different hardware and
    reports the result as though it were the microphone. That is precisely the
    failure being removed, so there is nothing to fall back to."""
    assert config._resolve_mixer_card(
        "Seiren", _cards_file(tmp_path, CARDS_NO_MIC)) is None


def test_returns_none_when_the_file_is_missing(tmp_path):
    assert config._resolve_mixer_card(
        "Seiren", str(tmp_path / "does_not_exist")) is None


def test_no_card_is_ever_guessed_on_a_miss(tmp_path):
    """Belt and braces on the two None cases above, stated as one property.

    Written as its own test because the tempting fix when this check starts
    reporting "cannot check" is to reintroduce a default, and a reader deleting
    one assertion should still trip the other."""
    for contents in (CARDS_NO_MIC, "", "garbage\n"):
        assert config._resolve_mixer_card(
            "Seiren", _cards_file(tmp_path, contents)) is None


# ── capsule identity ──
# The other half of "which microphone is this", and the same failure in a
# different place: a device reference that silently changes between boots.

def test_the_hardware_address_is_stripped():
    """PyAudio's name carries the ALSA card number, which shifts on reboot.

    voiceprint_samples recorded one physical Razer as three different
    microphones because of this, across ten samples, and the column exists
    precisely to keep two capsules from being averaged into one centroid."""
    assert (config.capsule_name("Razer Seiren V3 Mini: USB Audio (hw:3,0)")
            == "Razer Seiren V3 Mini: USB Audio")


def test_every_card_number_yields_the_same_capsule():
    """The property that matters, stated directly. The three strings below are
    the three that were actually in the table."""
    names = {config.capsule_name(f"Razer Seiren V3 Mini: USB Audio (hw:{n},0)")
             for n in (0, 1, 3)}
    assert len(names) == 1


def test_a_different_capsule_still_reads_as_different():
    """Stripping the address must not collapse genuinely different hardware,
    which is the failure that would matter once the array arrives."""
    assert (config.capsule_name("ReSpeaker 4 Mic Array (hw:1,0)")
            != config.capsule_name("Razer Seiren V3 Mini: USB Audio (hw:1,0)"))


def test_a_name_without_an_address_is_untouched():
    assert config.capsule_name("Some Mic") == "Some Mic"


def test_a_missing_name_does_not_raise():
    assert config.capsule_name(None) == ""
