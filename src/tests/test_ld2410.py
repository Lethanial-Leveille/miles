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


def test_engineering_reports_decode_the_same_leading_fields():
    # What the HLKRadarTool app leaves the module sending: the normal fields,
    # then max gates, nine moving and nine still energies, and two more bytes.
    data = (bytes([0x01, 0xAA, ld2410.BOTH]) + (166).to_bytes(2, "little") + bytes([61])
            + (300).to_bytes(2, "little") + bytes([63]) + b"\x00\x00"
            + bytes([8, 8]) + bytes(range(9)) + bytes(range(9)) + b"\x00\x00\x55\x00")
    assert len(data) == 35
    reports, _ = parse(_frame(data))
    assert reports == [Report(3, 166, 61, 300, 63)]


def test_an_unknown_report_type_is_skipped():
    unknown = _frame(bytes([0x07, 0xAA]) + bytes(11))
    reports, _ = parse(unknown + _normal(ld2410.MOVING, 100, 50, 0, 0))
    assert reports == [Report(1, 100, 50, 0, 0)]


def test_presence_is_any_state_but_nobody():
    assert not ld2410.is_present(Report(0, 444, 0, 444, 13))
    for state in (ld2410.MOVING, ld2410.STILL, ld2410.BOTH):
        assert ld2410.is_present(Report(state, 0, 0, 0, 0))
