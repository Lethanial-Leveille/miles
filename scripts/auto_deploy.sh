#!/usr/bin/env bash
# Deploy origin/main when it is not what the services are running, but only once its tests pass.
# Run by miles-deploy.timer. By hand: scripts/auto_deploy.sh --dry-run decides without restarting.

# Stop on any failed command (e), on an unset variable (u), and when any part of a pipe fails.
set -euo pipefail

# A fixed path, so the script behaves the same from any directory and under systemd.
REPO_DIR="/home/theycallmelee/miles"
# What deploy.sh writes after every service came up: the commit running now.
DEPLOYED_FILE="$REPO_DIR/data/deployed_commit"
# A commit whose deploy failed, so a crash on startup is not retried every two minutes.
FAILED_FILE="$REPO_DIR/data/deploy_failed_commit"
# The lock that keeps a manual run and a timer run from overlapping. build/ is RAM, fine for a lock.
LOCK_FILE="$REPO_DIR/build/auto_deploy.lock"
# Nova's database, read only, to see whether he is mid conversation.
DB_FILE="$REPO_DIR/data/miles.db"
# A turn that ended this recently means follow ups may still be coming.
QUIET_SECONDS=120

# Every line starts with this, so the journal shows which script spoke.
log() { echo "auto_deploy: $*"; }

# Off by default; --dry-run runs every check and the tests, then stops short of restarting.
DRY_RUN=0
# Only one optional argument is understood.
if [[ "${1:-}" == "--dry-run" ]]; then
    # Remember it for the last step.
    DRY_RUN=1
# Anything else is a typo worth stopping on.
elif [[ -n "${1:-}" ]]; then
    # Name the bad argument.
    echo "Usage: $0 [--dry-run]" >&2
    # Stop before touching anything.
    exit 1
fi

# Every git command below acts on this repo.
cd "$REPO_DIR"

# Open the lock file on descriptor 9; it stays open, and so locked, until the script exits.
exec 9>"$LOCK_FILE"
# -n gives up at once instead of waiting, because the other run is already doing this job.
if ! flock -n 9; then
    # Not an error: one deploy at a time is the point.
    log "another run holds the lock, skipping"
    # Zero, so systemd does not mark the unit failed.
    exit 0
fi

# Forget worktrees whose folders are gone, left behind if the Pi lost power mid test.
git worktree prune

# Without a record we cannot know what is running, and guessing would mean a restart for nothing.
if [[ ! -f "$DEPLOYED_FILE" ]]; then
    # Say how to create it: one deploy by hand writes it.
    log "no $DEPLOYED_FILE yet; run scripts/deploy.sh once by hand to create it"
    # Not a failure, just nothing safe to do.
    exit 0
fi
# The commit the services are running.
DEPLOYED="$(cat "$DEPLOYED_FILE")"
# The commit that failed last time, or empty when nothing has failed.
FAILED="$(cat "$FAILED_FILE" 2>/dev/null || true)"

# Update origin/main from GitHub. This changes no files in the repo, only git's copy of the remote.
git fetch --quiet origin main
# The commit GitHub has on main.
REMOTE="$(git rev-parse origin/main)"

# Already running what GitHub has: the normal case, so stay silent and leave the journal clean.
if [[ "$REMOTE" == "$DEPLOYED" ]]; then
    # Nothing to do.
    exit 0
fi
# This exact commit already failed; wait for a newer push instead of crashing production again.
if [[ "$REMOTE" == "$FAILED" ]]; then
    # One line, so it is visible why nothing is happening.
    log "${REMOTE:0:7} failed before; waiting for a newer push"
    # Not a new failure.
    exit 0
fi

# From here on there is something new to deploy. Say what.
log "origin/main is ${REMOTE:0:7}, services run ${DEPLOYED:0:7}"

# Uncommitted edits mean he is working on the Pi; a pull now could mix them with new code.
if [[ -n "$(git status --porcelain)" ]]; then
    # Retried next run, once the tree is clean.
    log "uncommitted changes in the repo, skipping until they are committed"
    # Not a failure.
    exit 0
fi
# Deploying from another branch would pull main into the wrong place.
if [[ "$(git rev-parse --abbrev-ref HEAD)" != "main" ]]; then
    # Retried once main is checked out again.
    log "not on main, skipping"
    # Not a failure.
    exit 0
fi
# deploy.sh pulls fast forward only, which needs the local commit to sit inside origin's history.
if ! git merge-base --is-ancestor HEAD origin/main; then
    # Local main has commits GitHub lacks, so the pull would refuse; pushing them fixes it.
    log "local main has commits origin lacks, skipping until they are pushed"
    # Not a failure.
    exit 0
fi

# The newest turn's end time, read only so this can never lock or change Nova's database.
LAST_TURN="$(python3 -c "
import sqlite3, sys
db = sqlite3.connect('file:' + sys.argv[1] + '?mode=ro', uri=True)
row = db.execute('SELECT created_at FROM timing_log ORDER BY id DESC LIMIT 1').fetchone()
print(row[0] if row else '')
" "$DB_FILE" 2>/dev/null || true)"
# An empty answer (no rows, or the read failed) is treated as quiet, so this gate cannot block forever.
if [[ -n "$LAST_TURN" ]]; then
    # Seconds since that turn; created_at is local time, and date reads it as local time too.
    AGE=$(( $(date +%s) - $(date -d "$LAST_TURN" +%s) ))
    # Recent enough that he may still be talking to her.
    if (( AGE < QUIET_SECONDS )); then
        # Retried in two minutes.
        log "a turn ended ${AGE}s ago, waiting for the conversation to finish"
        # Not a failure.
        exit 0
    fi
fi

# A fresh empty folder for the worktree, removed again below.
TEST_DIR="$(mktemp -d)"
# Whatever happens from here, take the worktree away again on exit.
trap 'git -C "$REPO_DIR" worktree remove --force "$TEST_DIR" 2>/dev/null || rm -rf "$TEST_DIR"' EXIT
# A second checkout of the new commit; ~/miles itself is untouched until the tests pass.
git worktree add --quiet --detach "$TEST_DIR" "$REMOTE"

# Announce the slow step.
log "testing ${REMOTE:0:7}"
# The suite exactly as the docs run it, from src/; the if keeps a failure from ending the script here.
if ! (cd "$TEST_DIR/src" && python3 -m pytest tests/ -q -p no:cacheprovider > "$TEST_DIR/pytest.log" 2>&1); then
    # The last lines hold the failing tests' names and the counts.
    tail -n 15 "$TEST_DIR/pytest.log" | sed 's/^/auto_deploy:   /'
    # Never retry this commit; a newer push gets its own chance.
    echo "$REMOTE" > "$FAILED_FILE"
    # Loud, so the journal says plainly that production was left alone.
    log "tests FAILED on ${REMOTE:0:7}; nothing restarted"
    # Non zero, so systemd marks the run failed.
    exit 1
fi
# The summary line, for the journal.
log "tests passed: $(tail -n 1 "$TEST_DIR/pytest.log")"

# A dry run stops here, after every check, with nothing changed.
if (( DRY_RUN )); then
    # Say what a real run would have done.
    log "dry run: would deploy ${REMOTE:0:7} now"
    # Done.
    exit 0
fi

# Pull, restart and report, the same as a deploy by hand. It writes DEPLOYED_FILE on success.
if ! "$REPO_DIR/scripts/deploy.sh"; then
    # A service did not come up; remember the commit so the next run does not restart it again.
    echo "$REMOTE" > "$FAILED_FILE"
    # deploy.sh already printed which service and its logs.
    log "deploy of ${REMOTE:0:7} FAILED; a service is not active, see the logs above"
    # Non zero, so systemd marks the run failed.
    exit 1
fi
# A success clears the old failure, so it cannot be mistaken for the current state.
rm -f "$FAILED_FILE"
# The line to look for in the journal.
log "deployed ${REMOTE:0:7}"
