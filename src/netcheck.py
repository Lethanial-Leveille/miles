"""Why did a network call fail? Answered locally, in milliseconds.

anthropic.APIConnectionError is one exception covering several very different
situations: the wifi dropped, the wifi is fine but nothing routes out, routing
is fine but DNS is dead, or everything on this end is healthy and the API
itself is unreachable. Telling him "I've lost wifi" when Anthropic is having an
outage is simply a false statement, and an assistant that guesses wrong about
its own state is worse than one that says less.

Nothing here reports a cause it did not observe. A socket failure carries its
reason in its errno, and reading that is the difference between four faults
that need four different responses from him:

  refused       something answered and shut the port. Route and name are fine.
  timed out     nothing answered at all. Firewall, dead upstream, black hole.
  no route      the kernel never sent a packet, having nowhere to send it.
  no resolution the name never became an address. DNS, and only DNS.

Collapsing those into "the network is down" sends him to power cycle a router
that was working, which is the same class of mistake as blaming wifi for an
Anthropic outage.

This only ever runs on a failure path, so its cost is paid by a turn that has
already gone wrong. The common case stays the cheapest: a dropped link is
answered by two file reads and never reaches a socket at all. The expensive
case is the one where everything local is healthy, and it costs one extra
PROBE_TIMEOUT to say what the API actually did instead of inferring it.
"""

import errno
import socket

# Probed by raw address on purpose. Reaching an IP without resolving a name is
# exactly what separates "nothing routes out" from "routing is fine, DNS is
# dead", and a hostname here would collapse the two.
INTERNET_PROBE = ("1.1.1.1", 443)
API_HOST       = "api.anthropic.com"
API_PORT       = 443
PROBE_TIMEOUT  = 1.5

# Probe outcomes, named rather than spelled out at each call site so a typo is
# an AttributeError here instead of a branch that silently never matches.
OK          = 'ok'
REFUSED     = 'refused'
TIMED_OUT   = 'timed_out'
NO_ROUTE    = 'no_route'
UNREACHABLE = 'unreachable'      # it failed, and the errno did not say how

# Every key diagnose can return. The test that each one has something to say
# reads this list, so adding a cause without giving Nova words for it fails in
# pytest rather than in the room, where the failure is silence.
CAUSES = ('no_wifi', 'no_route', 'net_timeout', 'no_internet',
          'no_dns', 'api_refused', 'api_timeout', 'api_down')


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


def probe(address, timeout=PROBE_TIMEOUT):
    """How a TCP connection to `address` failed, or OK if it did not.

    Order matters. TimeoutError and ConnectionRefusedError are both OSError
    subclasses, so the generic handler has to come last or it swallows the two
    specific answers this function exists to give.

    EHOSTUNREACH and ENETUNREACH are the kernel refusing to send rather than a
    peer refusing to answer, and ENETDOWN is the interface going away between
    the link check and here. All three mean the same thing to him: it never
    left the Pi."""
    try:
        socket.create_connection(address, timeout=timeout).close()
        return OK
    except TimeoutError:
        return TIMED_OUT
    except ConnectionRefusedError:
        return REFUSED
    except OSError as exc:
        if exc.errno in (errno.EHOSTUNREACH, errno.ENETUNREACH, errno.ENETDOWN):
            return NO_ROUTE
        return UNREACHABLE


def resolves(name):
    """Whether the resolver can turn `name` into an address.

    socket.gaierror only, not OSError. A resolver failure is the thing being
    tested here; anything else getaddrinfo raises is a bug on this end, and a
    bug should not reach him dressed as a DNS outage."""
    try:
        socket.getaddrinfo(name, API_PORT)
        return True
    except socket.gaierror:
        return False


def diagnose():
    """Return the phrase bank key describing why the network call failed.

    Ordered narrowest cause first, so each answer rules out everything above it
    and the phrase spoken is the most specific one that is actually true.

    Every branch below is something observed on this run, including api_down.
    It used to be whatever was left once the other three checks passed, which
    made it the one answer given with no evidence behind it, and it was given
    most often precisely when the truth was least obvious."""
    iface = default_interface()
    if iface is None or not link_up(iface):
        return 'no_wifi'

    outcome = probe(INTERNET_PROBE)
    if outcome == NO_ROUTE:
        return 'no_route'
    if outcome == TIMED_OUT:
        return 'net_timeout'
    if outcome != OK:
        # Reached by a refusal from a raw IP, which is a middlebox intercepting
        # rather than an honest path, and by any errno not listed above. Both
        # are real faults and neither is worth its own sentence.
        return 'no_internet'

    if not resolves(API_HOST):
        return 'no_dns'

    # Routing and resolution are both known good by the time this runs, so this
    # probe asks about the API and nothing else, and its answer is about the
    # API and nothing else.
    outcome = probe((API_HOST, API_PORT))
    if outcome == REFUSED:
        return 'api_refused'
    if outcome == TIMED_OUT:
        return 'api_timeout'

    # Includes OK, which is the honest odd one: the socket opened and the
    # request still failed, so the fault is above TCP and "the API is what is
    # unreachable" is as far as this module can narrow it.
    return 'api_down'
