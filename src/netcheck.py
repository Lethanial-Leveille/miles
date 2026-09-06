"""Why did a network call fail? Answered locally, in milliseconds.

anthropic.APIConnectionError is one exception covering several very different
situations: the wifi dropped, the wifi is fine but nothing routes out, routing
is fine but DNS is dead, or everything on this end is healthy and the API
itself is unreachable. Telling him "I've lost wifi" when Anthropic is having an
outage is simply a false statement, and an assistant that guesses wrong about
its own state is worse than one that says less.

Every check here is local. Nothing in this module can itself need the network
that is being questioned.

This only ever runs on a failure path, so its cost is paid by a turn that has
already gone wrong. The common case is also the cheapest: a dropped link is
answered by two file reads and never reaches a socket timeout.
"""

import socket

# Probed by raw address on purpose. Reaching an IP without resolving a name is
# exactly what separates "nothing routes out" from "routing is fine, DNS is
# dead", and a hostname here would collapse the two.
INTERNET_PROBE = ("1.1.1.1", 443)
API_HOST       = "api.anthropic.com"
PROBE_TIMEOUT  = 1.5


def default_interface(route_table="/proc/net/route"):
    """Interface holding the default route, or None if there is none.

    Read from /proc rather than shelling out to `ip`, and keyed on the default
    route rather than on link state, because Docker leaves bridge interfaces
    permanently UP. An "is any interface up" check calls a dead uplink healthy
    on this machine."""
    try:
        with open(route_table) as f:
            next(f, None)                       # column header
            for line in f:
                fields = line.split()
                # Iface Destination Gateway ... ; 00000000 is the default route
                if len(fields) > 1 and fields[1] == "00000000":
                    return fields[0]
    except OSError:
        return None
    return None


def link_up(iface, sysfs="/sys/class/net"):
    try:
        with open(f"{sysfs}/{iface}/operstate") as f:
            return f.read().strip() == "up"
    except OSError:
        return False


def reachable(address, timeout=PROBE_TIMEOUT):
    try:
        socket.create_connection(address, timeout=timeout).close()
        return True
    except OSError:
        return False


def resolves(name):
    try:
        socket.getaddrinfo(name, 443)
        return True
    except OSError:
        return False


def diagnose():
    """Return the phrase bank key describing why the network call failed.

    Ordered narrowest cause first, so each answer rules out everything above it
    and the phrase spoken is the most specific one that is actually true."""
    iface = default_interface()
    if iface is None or not link_up(iface):
        return 'no_wifi'
    if not reachable(INTERNET_PROBE):
        return 'no_internet'
    if not resolves(API_HOST):
        return 'no_dns'
    return 'api_down'
