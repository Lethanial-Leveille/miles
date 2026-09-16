#!/bin/sh
# Bring wlan0 back when it has dropped. Run every two minutes by
# miles-wifi.timer, never by hand except to test it.
#
# Why shell when everything else in scripts/ is Python: this reads two files
# and runs one command, and the boot where it matters most is the boot where
# the Python environment is itself the thing that is broken. There is nothing
# here worth an interpreter.
set -u

IFACE="wlan0"
PROFILE="Alsander"

# carrier returns EINVAL on an administratively down interface rather than a
# zero, so an unreadable value is a missing carrier and not a separate error
# worth reporting. Either way the answer is "bring it up".
carrier=$(cat "/sys/class/net/$IFACE/carrier" 2>/dev/null || echo 0)

# scope global, because a 169.254 link local address is exactly what a failed
# DHCP lease looks like, and counting it as an address is how a watchdog sleeps
# through the outage it was written for.
address=$(ip -4 -o addr show dev "$IFACE" scope global 2>/dev/null)

if [ "$carrier" = "1" ] && [ -n "$address" ]; then
    # Deliberately silent. systemd already writes a Starting and a Finished
    # line per run, so the heartbeat exists without this adding seven hundred
    # lines a day that all say nothing happened.
    exit 0
fi

echo "$IFACE is down: carrier=$carrier, address=${address:-none}. Bringing up $PROFILE."

# -w 30 so a failure is reported as a failure rather than hanging past the next
# timer fire and stacking two attempts against the same supplicant.
if nmcli -w 30 connection up "$PROFILE"; then
    echo "nmcli connection up $PROFILE succeeded."
else
    echo "nmcli connection up $PROFILE failed." >&2
    exit 1
fi
