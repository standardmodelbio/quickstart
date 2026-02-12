#!/bin/bash
# Tests for quickstart.sh uv install fix.
# Mocks curl, git, uv so nothing hits the network.
# Run: bash tests/test_quickstart.sh

PASS=0; FAIL=0
SCRIPT="$(cd "$(dirname "$0")/.." && pwd)/quickstart.sh"
ORIG_PATH="$PATH"
ORIG_HOME="$HOME"

assert() {
    local desc="$1"; shift
    if "$@"; then
        echo "  ✓ $desc"; ((PASS++))
    else
        echo "  ✗ $desc"; ((FAIL++))
    fi
}

setup_sandbox() {
    export PATH="$ORIG_PATH"
    export HOME="$ORIG_HOME"

    SANDBOX="$(mktemp -d)"
    export HOME="$SANDBOX/home"
    mkdir -p "$HOME/.local/bin" "$SANDBOX/work"

    MOCK_BIN="$SANDBOX/mock_bin"
    mkdir -p "$MOCK_BIN"

    # mock curl – outputs installer that creates fake uv + env file
    cat > "$MOCK_BIN/curl" << 'MOCK'
#!/bin/bash
cat << 'INSTALLER'
mkdir -p "$HOME/.local/bin"
printf '#!/bin/bash\necho "mock uv $@"\n' > "$HOME/.local/bin/uv"
chmod +x "$HOME/.local/bin/uv"
printf 'export PATH="$HOME/.local/bin:$PATH"\n' > "$HOME/.local/bin/env"
INSTALLER
MOCK
    chmod +x "$MOCK_BIN/curl"

    # mock git
    cat > "$MOCK_BIN/git" << 'MOCK'
#!/bin/bash
if [ "$1" = "clone" ]; then
    dir="${@: -1}"
    mkdir -p "$dir"
    echo '[project]' > "$dir/pyproject.toml"
fi
MOCK
    chmod +x "$MOCK_BIN/git"

    # Filtered system bin (essentials only, no real uv/curl/git)
    SYS_BIN="$SANDBOX/sys_bin"
    mkdir -p "$SYS_BIN"
    for cmd in sh bash cat echo printf grep chmod mkdir rm cp mv ln ls touch mktemp head tail sleep kill wait sed awk tr read env test tee sort wc; do
        real="$(command -v "$cmd" 2>/dev/null || true)"
        [ -n "$real" ] && ln -sf "$real" "$SYS_BIN/$cmd"
    done

    # Fake writable /usr/local/bin
    USR_LOCAL_BIN="$SANDBOX/usr_local_bin"
    mkdir -p "$USR_LOCAL_BIN"

    touch "$HOME/.bashrc" "$HOME/.profile"
    hash -r 2>/dev/null
    export PATH="$MOCK_BIN:$HOME/.local/bin:$USR_LOCAL_BIN:$SYS_BIN"
}

teardown_sandbox() {
    export PATH="$ORIG_PATH"
    export HOME="$ORIG_HOME"
    rm -rf "$SANDBOX"
}

run_script() {
    local patched="$SANDBOX/quickstart_patched.sh"
    sed "s|/usr/local/bin|$USR_LOCAL_BIN|g" "$SCRIPT" > "$patched"
    (cd "$SANDBOX/work" && echo "n" | bash "$patched") > "$SANDBOX/out.log" 2>&1 || true
}

# ─── Test 1: Fresh install – uv installed and symlinked ──────────
echo "Test 1: Fresh install – uv installed and symlinked"
setup_sandbox
rm -f "$HOME/.local/bin/uv"
run_script

assert "uv binary created"             test -x "$HOME/.local/bin/uv"
assert "uv symlinked to /usr/local/bin" test -L "$USR_LOCAL_BIN/uv"
assert "output shows uv found"         grep -q "uv found" "$SANDBOX/out.log"
teardown_sandbox

# ─── Test 2: Existing uv – no install, no symlink ────────────────
echo "Test 2: uv already on PATH – install skipped"
setup_sandbox
printf '#!/bin/bash\necho "mock uv $@"\n' > "$HOME/.local/bin/uv"
chmod +x "$HOME/.local/bin/uv"
run_script

assert "no symlink created" test ! -L "$USR_LOCAL_BIN/uv"
teardown_sandbox

# ─── Test 3: Existing uv elsewhere on PATH ───────────────────────
echo "Test 3: uv already at target – no reinstall, no overwrite"
setup_sandbox
rm -f "$HOME/.local/bin/uv"
printf '#!/bin/bash\necho "existing uv"\n' > "$USR_LOCAL_BIN/uv"
chmod +x "$USR_LOCAL_BIN/uv"
run_script

assert "install skipped"               test ! -e "$HOME/.local/bin/uv"
assert "existing binary not overwritten" test ! -L "$USR_LOCAL_BIN/uv"
teardown_sandbox

# ─── Test 4: Script uses resolved UV_BIN for sync ────────────────
echo "Test 4: UV_BIN resolved correctly"
setup_sandbox
rm -f "$HOME/.local/bin/uv"
run_script

assert "output shows resolved path" grep -q "uv found" "$SANDBOX/out.log"
teardown_sandbox

# ─── Summary ──────────────────────────────────────────────────────
echo ""
echo "Results: $PASS passed, $FAIL failed"
[ "$FAIL" -eq 0 ] && exit 0 || exit 1
