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
_REPORT_TYPES = (0x01, 0x02)
_DATA_HEAD = 0xAA

# An engineering report is 35 bytes. A length field far past that is corruption,
# and believing it would stall parsing waiting for bytes that never come.
_MAX_DATA_LEN = 64

NOBODY, MOVING, STILL, BOTH = 0, 1, 2, 3

Report = namedtuple("Report", "state moving_cm moving_energy still_cm still_energy")


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


def _decode(data):
    # A well framed report of a type this does not know is skipped, not guessed at.
    if len(data) < 11 or data[0] not in _REPORT_TYPES or data[1] != _DATA_HEAD:
        return None
    return Report(*struct.unpack_from("<BHBHB", data, 2))
