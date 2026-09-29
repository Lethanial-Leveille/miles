"""Read and tune the LD2410C presence radar over its UART, from the Pi.

Tuning used to mean the HLKRadarTool app over Bluetooth, which left the
module's settings in its flash and nowhere else. This does the same over the
wire the service reads, so the values can come from the repo.

miles-voice holds the port exclusively while it runs. Stop it first; the
presence log records the gap as silent, which is the truth.

Usage:
  sudo systemctl stop miles-voice
  python3 scripts/radar.py show            # the thresholds and range the module holds
  python3 scripts/radar.py record 120      # energy per gate for two minutes
  python3 scripts/radar.py record 120 --delay 30   # time to leave the room first
  python3 scripts/radar.py set 2 40 10     # gate 2: moving threshold 40, still 10
  python3 scripts/radar.py range 5 5       # count gates 0 to 5; nobody after 5s
  python3 scripts/radar.py resolution      # the gate size the module uses
  python3 scripts/radar.py resolution 0.2  # 20cm gates; restarts the module
  sudo systemctl start miles-voice
"""

import argparse
import os
import statistics
import sys
import time
from contextlib import contextmanager

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "src"))

import serial

import ld2410
from config import PRESENCE_BAUD, PRESENCE_PORT

# A restarted module takes a moment before it answers again.
RESTART_WAIT_S = 3.0

def read_params(port):
    ack = ld2410.send(port, ld2410.command(ld2410.READ_PARAMS), ld2410.READ_PARAMS)
    params = ld2410.decode_params(ack.data) if ack.ok else None
    if params is None:
        raise RuntimeError("the radar's parameter reply did not decode")
    return params


def read_gate_cm(port):
    """The gate size in centimetres, as the module reports it, so every distance
    printed matches what the gates actually cover."""
    ack = ld2410.send(port, ld2410.command(ld2410.READ_RESOLUTION), ld2410.READ_RESOLUTION)
    code = ld2410.decode_resolution(ack.data) if ack.ok else None
    if code not in ld2410.GATE_CM:
        raise RuntimeError(f"the radar reported an unknown resolution: {code}")
    return ld2410.GATE_CM[code]


def print_params(params, gate_cm):
    print(f"Range: moving to gate {params.max_moving_gate}, still to gate "
          f"{params.max_still_gate}. Nobody after {params.unmanned_s}s.\n")
    print("gate   distance   moving   still")
    for gate in range(params.max_gate + 1):
        counted = gate <= max(params.max_moving_gate, params.max_still_gate)
        note = "" if counted else "   (out of range)"
        print(f"{gate:>4}   {gate * gate_cm:>5} cm   {params.moving[gate]:>6}   "
              f"{params.still[gate]:>5}{note}")


@contextmanager
def engineering_mode(port):
    """Energy per gate in every report, restored to normal reports afterwards.
    The module drops it on power loss anyway; this keeps a Ctrl+C from leaving
    it on for the service."""
    with ld2410.config_mode(port):
        ld2410.send(port, ld2410.command(ld2410.ENGINEERING_ON), ld2410.ENGINEERING_ON)
    try:
        yield
    finally:
        with ld2410.config_mode(port):
            ld2410.send(port, ld2410.command(ld2410.ENGINEERING_OFF), ld2410.ENGINEERING_OFF)


def collect(port, seconds):
    """Engineering reports for this long. Reports without gate data are dropped."""
    reports, buf = [], b""
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        got, buf = ld2410.parse(buf + port.read(64))
        reports += [r for r in got if r.moving_gates is not None]
    return reports


def _spread(values):
    return f"{min(values):>3} {statistics.median(values):>5.0f} {max(values):>3}"


def print_energies(reports, params, gate_cm):
    """Lowest, median and highest energy per gate, beside the threshold it is
    compared against. A threshold belongs above everything an empty room
    produces and below everything a person does."""
    present = sum(ld2410.is_present(r) for r in reports)
    print(f"{len(reports)} reports, module said someone was there in "
          f"{100 * present / len(reports):.0f}% of them.\n")
    print("                  moving energy              still energy")
    print("gate  distance   min  med max  thresh     min  med max  thresh")
    gates = min(len(reports[0].moving_gates), len(reports[0].still_gates))
    for gate in range(gates):
        moving = [r.moving_gates[gate] for r in reports]
        still = [r.still_gates[gate] for r in reports]
        print(f"{gate:>4}  {gate * gate_cm:>5} cm   {_spread(moving)}  {params.moving[gate]:>6}"
              f"     {_spread(still)}  {params.still[gate]:>6}")


def open_port():
    try:
        return serial.Serial(PRESENCE_PORT, PRESENCE_BAUD, timeout=0.2, exclusive=True)
    except serial.SerialException as exc:
        sys.exit(f"Cannot open {PRESENCE_PORT}: {exc}\n"
                 "If miles-voice is running it holds the port: "
                 "sudo systemctl stop miles-voice")


def cmd_show(args):
    with open_port() as port, ld2410.config_mode(port):
        gate_cm = read_gate_cm(port)
        print(f"Gates are {gate_cm} cm.")
        print_params(read_params(port), gate_cm)


def cmd_record(args):
    with open_port() as port:
        with ld2410.config_mode(port):
            params = read_params(port)
            gate_cm = read_gate_cm(port)
        if args.delay:
            print(f"Starting in {args.delay}s.", flush=True)
            time.sleep(args.delay)
        print(f"Recording {args.seconds}s...", flush=True)
        with engineering_mode(port):
            reports = collect(port, args.seconds)
    if not reports:
        sys.exit("No engineering reports arrived.")
    print_energies(reports, params, gate_cm)


def _write(port, frame, command):
    """Send a setting, then read everything back, so what is printed is what the
    module holds rather than what was asked for."""
    if not ld2410.send(port, frame, command).ok:
        sys.exit("The radar refused the setting.")
    print_params(read_params(port), read_gate_cm(port))


def cmd_set(args):
    if not 0 <= args.gate <= 8 or not all(0 <= t <= 100 for t in (args.moving, args.still)):
        sys.exit("Gates run 0 to 8 and thresholds 0 to 100.")
    with open_port() as port, ld2410.config_mode(port):
        _write(port, ld2410.set_sensitivity(args.gate, args.moving, args.still),
               ld2410.SET_SENSITIVITY)


def cmd_range(args):
    if not 1 <= args.gate <= 8:
        sys.exit("The farthest gate runs 1 to 8.")
    with open_port() as port, ld2410.config_mode(port):
        _write(port, ld2410.set_max_gates(args.gate, args.gate, args.unmanned),
               ld2410.SET_MAX_GATES)


def cmd_resolution(args):
    with open_port() as port:
        if args.size is None:
            with ld2410.config_mode(port):
                print(f"Gates are {read_gate_cm(port)} cm.")
            return
        code = {0.2: 1, 0.75: 0}.get(args.size)
        if code is None:
            sys.exit("The module offers 0.2 or 0.75.")
        # The module answers RESTART and then reboots, so it never takes the
        # END_CONFIG that config_mode would send on the way out. Entered by hand.
        if not ld2410.send(port, ld2410.enable_config(), ld2410.ENABLE_CONFIG).ok:
            sys.exit("The radar refused to enter config mode.")
        if not ld2410.send(port, ld2410.set_resolution(code), ld2410.SET_RESOLUTION).ok:
            sys.exit("The radar refused the resolution.")
        ld2410.send(port, ld2410.command(ld2410.RESTART), ld2410.RESTART)
        time.sleep(RESTART_WAIT_S)
        port.reset_input_buffer()
        with ld2410.config_mode(port):
            gate_cm = read_gate_cm(port)
            print(f"Restarted. Gates are {gate_cm} cm.")
            print_params(read_params(port), gate_cm)


def main():
    parser = argparse.ArgumentParser(description="Read and tune the presence radar.")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("show", help="the thresholds and range the module holds")
    record = sub.add_parser("record", help="energy per gate over a stretch of time")
    record.add_argument("seconds", type=float)
    record.add_argument("--delay", type=float, default=0,
                        help="seconds to wait first, to leave the room")
    setter = sub.add_parser("set", help="one gate's moving and still thresholds")
    setter.add_argument("gate", type=int)
    setter.add_argument("moving", type=int)
    setter.add_argument("still", type=int)
    ranger = sub.add_parser("range", help="the farthest gate that counts, and the unmanned duration")
    ranger.add_argument("gate", type=int)
    ranger.add_argument("unmanned", type=int, help="seconds after the last detection")
    resolution = sub.add_parser("resolution", help="read or set the gate size, in metres")
    resolution.add_argument("size", type=float, nargs="?")
    args = parser.parse_args()
    {"show": cmd_show, "record": cmd_record, "set": cmd_set,
     "range": cmd_range, "resolution": cmd_resolution}[args.command](args)


if __name__ == "__main__":
    main()
