"""The LD2410C presence radar's report frames, decoded from its serial stream.

The UART delivers bytes with no boundaries: one read can end halfway through a
frame or hold one and a half. So parse() takes a buffer, returns every complete
report in it, and hands back the tail for the next read to extend.

Pure bytes in and out, never the port, so it is tested without the hardware.
Wiring, tuning and why the module is set the way it is: BACKEND_TODO.md,
Presence detection.
"""

import struct
from collections import namedtuple

HEADER = b"\xf4\xf3\xf2\xf1"
FOOTER = b"\xf8\xf7\xf6\xf5"

# Report type 0x02 is normal, 0x01 is engineering mode, which appends energy
# per gate. The HLKRadarTool app turns engineering mode on, so both arrive in
# practice and the leading fields are laid out the same in each.
_ENGINEERING = 0x01
_REPORT_TYPES = (_ENGINEERING, 0x02)
_DATA_HEAD = 0xAA

# An engineering report is 35 bytes. A length field far past that is corruption,
# and believing it would stall parsing waiting for bytes that never come.
_MAX_DATA_LEN = 64

NOBODY, MOVING, STILL, BOTH = 0, 1, 2, 3

# The gate lists are only in engineering reports: the energy the module saw in
# each gate, which is what a threshold is compared against. None otherwise.
Report = namedtuple("Report", "state moving_cm moving_energy still_cm still_energy "
                              "moving_gates still_gates", defaults=(None, None))


def is_present(report):
    """The module's own verdict. Its thresholds and range decide it, not ours."""
    return report.state != NOBODY


def parse(buf):
    """Every complete report in buf, and the bytes left over.

    A frame whose footer is wrong was never a frame: the header was a chance
    match inside garbage, from a loose wire or a read that started mid frame.
    Skip one byte and look again rather than trusting the length it claimed.
    """
    reports = []
    while True:
        start = buf.find(HEADER)
        if start < 0:
            # A header can be split across two reads, so keep enough tail
            # to complete one.
            return reports, buf[-(len(HEADER) - 1):]
        buf = buf[start:]
        if len(buf) < len(HEADER) + 2:
            return reports, buf

        length = struct.unpack_from("<H", buf, len(HEADER))[0]
        if length > _MAX_DATA_LEN:
            buf = buf[1:]
            continue

        data_start = len(HEADER) + 2
        end = data_start + length + len(FOOTER)
        if len(buf) < end:
            return reports, buf
        if buf[end - len(FOOTER):end] != FOOTER:
            buf = buf[1:]
            continue

        report = _decode(buf[data_start:data_start + length])
        if report is not None:
            reports.append(report)
        buf = buf[end:]


# ── Commands ──
# The same UART takes commands, framed like reports but with their own header
# and footer. Every command gets an acknowledgement. They are only accepted
# between ENABLE_CONFIG and END_CONFIG, and the module sends no reports in
# between, which is the silence seen whenever the HLKRadarTool app connects.

CMD_HEADER = b"\xfd\xfc\xfb\xfa"
CMD_FOOTER = b"\x04\x03\x02\x01"

ENABLE_CONFIG = 0x00FF
END_CONFIG = 0x00FE
SET_MAX_GATES = 0x0060
READ_PARAMS = 0x0061
ENGINEERING_ON = 0x0062
ENGINEERING_OFF = 0x0063
SET_SENSITIVITY = 0x0064

# The acknowledgement echoes the command word with this bit set.
_ACK_BIT = 0x0100

Ack = namedtuple("Ack", "command ok data")
Params = namedtuple("Params", "max_gate max_moving_gate max_still_gate "
                              "moving still unmanned_s")


def command(word, value=b""):
    body = struct.pack("<H", word) + value
    return CMD_HEADER + struct.pack("<H", len(body)) + body + CMD_FOOTER


def enable_config():
    return command(ENABLE_CONFIG, struct.pack("<H", 1))


def set_sensitivity(gate, moving, still):
    """One gate's thresholds, 0 to 100. A reading above it counts as someone."""
    return command(SET_SENSITIVITY, struct.pack("<HIHIHI", 0, gate, 1, moving, 2, still))


def set_max_gates(moving_gate, still_gate, unmanned_s):
    """The farthest gate that counts, for each kind of target, and how long
    after the last detection the module reports nobody."""
    return command(SET_MAX_GATES,
                   struct.pack("<HIHIHI", 0, moving_gate, 1, still_gate, 2, unmanned_s))


def parse_acks(buf):
    """Every complete acknowledgement in buf, and the bytes left over.

    The same resync rule as parse(): a wrong footer means the header was a
    chance match, so move one byte on."""
    acks = []
    while True:
        start = buf.find(CMD_HEADER)
        if start < 0:
            return acks, buf[-(len(CMD_HEADER) - 1):]
        buf = buf[start:]
        if len(buf) < len(CMD_HEADER) + 2:
            return acks, buf

        length = struct.unpack_from("<H", buf, len(CMD_HEADER))[0]
        if length > _MAX_DATA_LEN or length < 4:
            buf = buf[1:]
            continue

        data_start = len(CMD_HEADER) + 2
        end = data_start + length + len(CMD_FOOTER)
        if len(buf) < end:
            return acks, buf
        if buf[end - len(CMD_FOOTER):end] != CMD_FOOTER:
            buf = buf[1:]
            continue

        word, status = struct.unpack_from("<HH", buf, data_start)
        acks.append(Ack(word & ~_ACK_BIT, status == 0, buf[data_start + 4:end - len(CMD_FOOTER)]))
        buf = buf[end:]


def decode_params(data):
    """The READ_PARAMS acknowledgement: range, both thresholds per gate, and
    the unmanned duration."""
    if len(data) < 4 or data[0] != _DATA_HEAD:
        return None
    max_gate, max_moving, max_still = data[1], data[2], data[3]
    gates = max_gate + 1
    moving = list(data[4:4 + gates])
    still = list(data[4 + gates:4 + 2 * gates])
    unmanned_s = struct.unpack_from("<H", data, 4 + 2 * gates)[0]
    return Params(max_gate, max_moving, max_still, moving, still, unmanned_s)


def _decode(data):
    # A well framed report of a type this does not know is skipped, not guessed at.
    if len(data) < 11 or data[0] not in _REPORT_TYPES or data[1] != _DATA_HEAD:
        return None
    fields = struct.unpack_from("<BHBHB", data, 2)
    if data[0] != _ENGINEERING or len(data) < 13:
        return Report(*fields)

    # After the target fields: the highest moving gate and highest still gate,
    # then one energy byte per gate for each, zero upward.
    moving_count, still_count = data[11] + 1, data[12] + 1
    moving = list(data[13:13 + moving_count])
    still = list(data[13 + moving_count:13 + moving_count + still_count])
    if len(moving) != moving_count or len(still) != still_count:
        return Report(*fields)
    return Report(*fields, moving, still)
