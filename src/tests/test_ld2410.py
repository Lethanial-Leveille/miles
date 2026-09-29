import ld2410
from ld2410 import Report, parse

# Three reports exactly as the sensor on the desk sent them on Sep 27 2026, the
# first read after wiring it: someone still at 30cm, energy 14 then 13.
FIRST_CAPTURE = bytes.fromhex(
    "f4f3f2f1 0d00 02aa 02 1e00 00 1e00 0e 0000 5500 f8f7f6f5"
    "f4f3f2f1 0d00 02aa 02 1e00 00 1e00 0e 0000 5500 f8f7f6f5"
    "f4f3f2f1 0d00 02aa 02 1e00 00 1e00 0d 0000 55")


def _frame(data):
    return ld2410.HEADER + len(data).to_bytes(2, "little") + data + ld2410.FOOTER


def _normal(state, moving_cm, moving_energy, still_cm, still_energy):
    data = (bytes([0x02, 0xAA, state]) + moving_cm.to_bytes(2, "little")
            + bytes([moving_energy]) + still_cm.to_bytes(2, "little")
            + bytes([still_energy]) + b"\x00\x00\x55\x00")
    return _frame(data)


def test_real_capture_decodes_and_keeps_the_partial_frame():
    reports, rest = parse(FIRST_CAPTURE)
    assert reports == [Report(2, 30, 0, 30, 14), Report(2, 30, 0, 30, 14)]
    # The third frame was cut off by the read; the next read completes it.
    assert rest.startswith(ld2410.HEADER)
    reports, rest = parse(rest + b"\x00" + ld2410.FOOTER)
    assert reports == [Report(2, 30, 0, 30, 13)]
    assert rest == b""


def test_a_frame_split_across_reads_is_completed_by_the_next():
    frame = _normal(ld2410.MOVING, 438, 21, 468, 20)
    reports, rest = parse(frame[:9])
    assert reports == []
    reports, rest = parse(rest + frame[9:])
    assert reports == [Report(1, 438, 21, 468, 20)]
    assert rest == b""


def test_a_header_split_across_reads_survives():
    frame = _normal(ld2410.NOBODY, 0, 0, 0, 0)
    reports, rest = parse(b"\x11\x22" + frame[:2])
    assert reports == []
    reports, _ = parse(rest + frame[2:])
    assert reports == [Report(0, 0, 0, 0, 0)]


def test_garbage_before_a_frame_is_skipped():
    reports, _ = parse(b"\x00\xff\x13\x37" + _normal(ld2410.STILL, 0, 0, 162, 100))
    assert reports == [Report(2, 0, 0, 162, 100)]


def test_a_wrong_footer_resyncs_instead_of_trusting_the_length():
    good = _normal(ld2410.STILL, 0, 0, 380, 100)
    broken = good[:-1] + b"\x00"
    reports, _ = parse(broken + good)
    assert reports == [Report(2, 0, 0, 380, 100)]


def test_an_absurd_length_does_not_stall_parsing():
    bogus = ld2410.HEADER + (60000).to_bytes(2, "little")
    good = _normal(ld2410.NOBODY, 444, 0, 444, 13)
    reports, _ = parse(bogus + good)
    assert reports == [Report(0, 444, 0, 444, 13)]


def test_engineering_reports_decode_the_energy_in_every_gate():
    # What the HLKRadarTool app leaves the module sending: the normal fields,
    # then max gates, nine moving and nine still energies, and two more bytes.
    data = (bytes([0x01, 0xAA, ld2410.BOTH]) + (166).to_bytes(2, "little") + bytes([61])
            + (300).to_bytes(2, "little") + bytes([63]) + b"\x00\x00"
            + bytes([8, 8]) + bytes(range(9)) + bytes(range(9)) + b"\x00\x00\x55\x00")
    assert len(data) == 35
    reports, _ = parse(_frame(data))
    assert reports == [Report(3, 166, 61, 300, 63, list(range(9)), list(range(9)))]


def test_an_unknown_report_type_is_skipped():
    unknown = _frame(bytes([0x07, 0xAA]) + bytes(11))
    reports, _ = parse(unknown + _normal(ld2410.MOVING, 100, 50, 0, 0))
    assert reports == [Report(1, 100, 50, 0, 0)]


def test_presence_is_any_state_but_nobody():
    assert not ld2410.is_present(Report(0, 444, 0, 444, 13))
    for state in (ld2410.MOVING, ld2410.STILL, ld2410.BOTH):
        assert ld2410.is_present(Report(state, 0, 0, 0, 0))


# ── Commands ──
# Byte for byte the example frames in Hi Link's LD2410 serial protocol manual.

def test_command_frames_match_the_manual():
    assert ld2410.enable_config() == bytes.fromhex("fdfcfbfa 0400 ff00 0100 04030201")
    assert ld2410.command(ld2410.END_CONFIG) == bytes.fromhex("fdfcfbfa 0200 fe00 04030201")
    assert ld2410.command(ld2410.READ_PARAMS) == bytes.fromhex("fdfcfbfa 0200 6100 04030201")


def test_set_sensitivity_matches_the_manual():
    # The manual's example: every gate (0xFFFF) to 40 moving and 40 still.
    assert ld2410.set_sensitivity(0xFFFF, 40, 40) == bytes.fromhex(
        "fdfcfbfa 1400 6400 0000 ffff0000 0100 28000000 0200 28000000 04030201")


def test_set_max_gates_lays_out_three_words():
    frame = ld2410.set_max_gates(5, 5, 5)
    assert frame == bytes.fromhex(
        "fdfcfbfa 1400 6000 0000 05000000 0100 05000000 0200 05000000 04030201")


def test_ack_reports_the_command_and_success():
    acks, rest = ld2410.parse_acks(bytes.fromhex(
        "fdfcfbfa 0800 ff01 0000 0100 4000 04030201"))
    assert acks == [ld2410.Ack(ld2410.ENABLE_CONFIG, True, bytes.fromhex("01004000"))]
    assert rest == b""


def test_a_failed_ack_is_not_ok():
    acks, _ = ld2410.parse_acks(bytes.fromhex("fdfcfbfa 0400 6401 0100 04030201"))
    assert acks == [ld2410.Ack(ld2410.SET_SENSITIVITY, False, b"")]


def test_acks_are_found_among_reports():
    # Reports can still be arriving when the first command lands.
    report = _normal(ld2410.STILL, 0, 0, 145, 83)
    ack = bytes.fromhex("fdfcfbfa 0400 fe01 0000 04030201")
    acks, _ = ld2410.parse_acks(report + ack)
    assert acks == [ld2410.Ack(ld2410.END_CONFIG, True, b"")]


def test_factory_parameters_decode():
    # The manual's READ_PARAMS reply, which is also the factory default.
    acks, _ = ld2410.parse_acks(bytes.fromhex(
        "fdfcfbfa 1c00 6101 0000 aa 08 08 08"
        " 3232281e140f0f0f0f 000028281e1e141414 0500 04030201"))
    params = ld2410.decode_params(acks[0].data)
    assert params == ld2410.Params(
        8, 8, 8,
        [50, 50, 40, 30, 20, 15, 15, 15, 15],
        [0, 0, 40, 40, 30, 30, 20, 20, 20],
        5)


def test_resolution_frames():
    assert ld2410.set_resolution(1) == bytes.fromhex("fdfcfbfa 0400 aa00 0100 04030201")
    assert ld2410.command(ld2410.READ_RESOLUTION) == bytes.fromhex("fdfcfbfa 0200 ab00 04030201")
    acks, _ = ld2410.parse_acks(bytes.fromhex("fdfcfbfa 0600 ab01 0000 0100 04030201"))
    assert acks[0].command == ld2410.READ_RESOLUTION
    assert ld2410.GATE_CM[ld2410.decode_resolution(acks[0].data)] == 20
