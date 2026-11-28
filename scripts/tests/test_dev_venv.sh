#!/bin/sh
set -eu

root=$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd -P)
sandbox=$(mktemp -d)
trap 'rm -rf "$sandbox"' EXIT
checkout="$sandbox/checkout"
mkdir -p "$checkout/scripts" "$sandbox/tools" "$sandbox/peer/bin"
cp "$root/scripts/dev" "$checkout/scripts/dev"
touch "$sandbox/tools/python3.13"
chmod +x "$sandbox/tools/python3.13"

cat > "$sandbox/tools/uv" <<'EOF'
#!/bin/sh
touch "$UV_PROJECT_ENVIRONMENT/../sync-called"
EOF
cat > "$sandbox/peer/bin/python" <<EOF
#!/bin/sh
touch "$checkout/python-called"
EOF
chmod +x "$sandbox/tools/uv" "$sandbox/peer/bin/python"
export PATH="$sandbox/tools:$PATH"

ln -s "$sandbox/peer" "$checkout/.venv"
for command in bootstrap check; do
    if "$checkout/scripts/dev" "$command" >"$sandbox/result" 2>&1; then
        printf '%s\n' "Launcher accepted a symlinked environment directory for $command" >&2
        exit 1
    fi
    grep -q 'refuses a symlinked virtual environment directory' "$sandbox/result"
    test ! -e "$checkout/sync-called"
    test ! -e "$checkout/python-called"
done

rm "$checkout/.venv"
mkdir -p "$checkout/.venv/bin"
ln -s "$sandbox/peer/bin/python" "$checkout/.venv/bin/python"
"$checkout/scripts/dev" bootstrap
test -f "$checkout/sync-called"
test -f "$checkout/python-called"
printf '%s\n' 'Launcher refused the directory alias and accepted an interpreter symlink'
