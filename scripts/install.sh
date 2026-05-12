#!/usr/bin/env bash
set -euo pipefail

# Titan one-shot installer for macOS/Linux.
# Usage from a checkout: ./scripts/install.sh
# Usage from GitHub: curl -fsSL https://raw.githubusercontent.com/austindixson/Titan/main/scripts/install.sh | bash
# Override repo: TITAN_REPO_URL=https://github.com/OWNER/REPO.git bash scripts/install.sh

REPO_URL="${TITAN_REPO_URL:-https://github.com/austindixson/Titan.git}"
INSTALL_ROOT="${TITAN_INSTALL_ROOT:-$HOME/.titan}"
VENV_DIR="$INSTALL_ROOT/venv"
BIN_DIR="$HOME/.local/bin"
PYTHON_BIN="${PYTHON:-}"

log() { printf '[titan-install] %s\n' "$*"; }
fail() { printf '[titan-install] ERROR: %s\n' "$*" >&2; exit 1; }

if [ -z "$PYTHON_BIN" ]; then
  if command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="python3"
  elif command -v python >/dev/null 2>&1; then
    PYTHON_BIN="python"
  else
    fail "Python 3.10+ is required. Install it from https://python.org/downloads/ and rerun."
  fi
fi

"$PYTHON_BIN" - <<'PY' || fail "Python 3.10+ is required."
import sys
raise SystemExit(0 if sys.version_info >= (3, 10) else 1)
PY

mkdir -p "$INSTALL_ROOT" "$BIN_DIR"
log "creating virtual environment at $VENV_DIR"
"$PYTHON_BIN" -m venv "$VENV_DIR"
# shellcheck disable=SC1091
. "$VENV_DIR/bin/activate"
python -m pip install --upgrade pip

if [ -f "pyproject.toml" ] && [ -d "src/titan" ]; then
  log "installing Titan from current checkout"
  python -m pip install -e .
else
  log "installing Titan from $REPO_URL"
  python -m pip install "git+$REPO_URL"
fi

cat > "$BIN_DIR/titan" <<EOF
#!/usr/bin/env bash
exec "$VENV_DIR/bin/titan" "\$@"
EOF
chmod +x "$BIN_DIR/titan"

cat > "$BIN_DIR/titan-tui" <<EOF
#!/usr/bin/env bash
exec "$VENV_DIR/bin/titan-tui" "\$@"
EOF
chmod +x "$BIN_DIR/titan-tui"

"$BIN_DIR/titan" setup || true
"$BIN_DIR/titan" config set chat_recaps_enabled false >/dev/null || true

log "installed Titan"
log "launch: $BIN_DIR/titan"
log "optional legacy launcher: $BIN_DIR/titan-tui"
case ":$PATH:" in
  *":$BIN_DIR:"*) ;;
  *) log "add this to your shell profile if needed: export PATH=\"$BIN_DIR:\$PATH\"" ;;
esac
log "next: titan doctor"
