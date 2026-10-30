#!/usr/bin/env bash
# Deploy the latest main on the Pi and restart the MILES services.
# Usage: scripts/deploy.sh          restarts miles-voice and miles-server
#        scripts/deploy.sh tunnel   also restarts miles-tunnel

# Stop on any failed command (e), on an unset variable (u), and when any part of a pipe fails.
set -euo pipefail

# A fixed path, so the script behaves the same from any directory.
REPO_DIR="/home/theycallmelee/miles"

# The tunnel is left out by default: restarting it drops the app's connection for a few seconds.
SERVICES=(miles-voice miles-server)

# Seconds to wait after the restart, so a service that crashes on startup has time to show it.
SETTLE_SECONDS=3

# Only one optional argument is understood; anything else is a typo worth stopping on.
if [[ $# -gt 1 ]]; then
    # Say what went wrong on stderr, where errors belong.
    echo "Usage: $0 [tunnel]" >&2
    # A non zero exit tells the shell the deploy did not happen.
    exit 1
fi

# ${1:-} reads the first argument, or an empty string when none was given (set -u would fail otherwise).
if [[ "${1:-}" == "tunnel" ]]; then
    # Add the tunnel to the restart list only when asked for by name.
    SERVICES+=(miles-tunnel)
# Any argument other than "tunnel" is refused rather than silently ignored.
elif [[ -n "${1:-}" ]]; then
    # Name the bad argument so the mistake is obvious.
    echo "Unknown argument: $1 (the only option is: tunnel)" >&2
    # Stop before touching anything.
    exit 1
fi

# Every git command below acts on this repo.
cd "$REPO_DIR"

# --porcelain prints one line per modified, staged or untracked file, and nothing when the tree is clean.
if [[ -n "$(git status --porcelain)" ]]; then
    # Explain why the deploy stopped.
    echo "Uncommitted changes in $REPO_DIR. Commit or stash them first:" >&2
    # Show which files are dirty, in git's short format.
    git status --short >&2
    # A pull on top of local edits could mix them with new code, so refuse.
    exit 1
fi

# The name of the branch currently checked out.
BRANCH="$(git rev-parse --abbrev-ref HEAD)"
# Deploying from another branch would pull main into the wrong place.
if [[ "$BRANCH" != "main" ]]; then
    # Name the branch so it is clear what to switch away from.
    echo "On branch $BRANCH, not main. Run: git checkout main" >&2
    # Stop without pulling.
    exit 1
fi

# The commit running now, kept so the summary can say what changed.
OLD_COMMIT="$(git rev-parse HEAD)"

# Announce the step, since a network fetch can take a moment.
echo "Pulling origin main (fast forward only)..."
# --ff-only refuses to create a merge commit; if local and origin have diverged it fails and set -e stops the script.
git pull --ff-only origin main

# The commit that will run after the restart.
NEW_COMMIT="$(git rev-parse HEAD)"

# The same hash before and after means nothing new came down.
if [[ "$OLD_COMMIT" == "$NEW_COMMIT" ]]; then
    # Still restart below, so the script can also bring back a service that fell over.
    echo "Already up to date at ${NEW_COMMIT:0:7}."
else
    # Short hashes, the first seven characters, are enough to tell commits apart.
    echo "Updated ${OLD_COMMIT:0:7} to ${NEW_COMMIT:0:7}:"
    # One line per new commit, so you can see what is about to go live.
    git log --oneline "$OLD_COMMIT..$NEW_COMMIT"
fi

# Dependencies are installed by hand for now; this line keeps that from being forgotten.
echo "Reminder: Python dependencies are not managed by this script. Install any new ones by hand."

# Show which services are about to restart; "${SERVICES[*]}" joins the list with spaces.
echo "Restarting: ${SERVICES[*]}"
# Restart them all in one call; "${SERVICES[@]}" passes each name as its own argument.
sudo systemctl restart "${SERVICES[@]}"

# Give each service a moment to either settle or crash.
sleep "$SETTLE_SECONDS"

# Collect the services that did not come up, for the summary at the end.
FAILED=()

# Report on each service in turn.
for SERVICE in "${SERVICES[@]}"; do
    # A blank line and a header keep each service's output separate.
    echo
    # The header names the service being shown.
    echo "========== $SERVICE =========="
    # --no-pager prints straight to the terminal; || true because status exits non zero for an inactive service.
    systemctl status "$SERVICE" --no-pager || true
    # A divider between the status block and the logs.
    echo "---------- last 20 log lines ----------"
    # sudo so the journal is readable whatever groups this user is in; -n 20 is the last twenty lines.
    sudo journalctl -u "$SERVICE" -n 20 --no-pager
    # is-active --quiet prints nothing and exits zero only when the service is running.
    if ! systemctl is-active --quiet "$SERVICE"; then
        # Remember the failure rather than stopping, so every service still gets reported.
        FAILED+=("$SERVICE")
    fi
done

# A blank line before the summary.
echo
# An empty list means every restarted service is running.
if [[ ${#FAILED[@]} -eq 0 ]]; then
    # One line that answers the question at a glance.
    echo "All services active: ${SERVICES[*]} at ${NEW_COMMIT:0:7}"
else
    # Name the ones that did not start, so you know whose logs to read above.
    echo "NOT active: ${FAILED[*]}" >&2
    # Non zero, so the deploy counts as failed if anything else ever runs this script.
    exit 1
fi
