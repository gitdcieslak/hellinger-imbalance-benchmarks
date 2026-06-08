#!/usr/bin/env bash
set -euo pipefail

BRANCH="${BRANCH:-research/accessibility-next-experiments}"
REPO_URL="${REPO_URL:-}"
REPO_DIR="${REPO_DIR:-}"

run_sudo() {
  if [ "$(id -u)" -eq 0 ]; then
    "$@"
  else
    sudo "$@"
  fi
}

echo "==> Installing system packages"
run_sudo apt update
run_sudo apt install -y git curl build-essential python3 python3-venv htop tmux rsync

echo "==> Installing uv"
if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export PATH="$HOME/.local/bin:$PATH"

if [ -n "$REPO_URL" ]; then
  if [ -z "$REPO_DIR" ]; then
    repo_name="$(basename "$REPO_URL")"
    REPO_DIR="${repo_name%.git}"
  fi
  if [ ! -d "$REPO_DIR/.git" ]; then
    echo "==> Cloning $REPO_URL into $REPO_DIR"
    git clone "$REPO_URL" "$REPO_DIR"
  fi
  cd "$REPO_DIR"
fi

echo "==> Checking out branch $BRANCH"
git fetch --all --prune || true
git checkout "$BRANCH"

echo "==> Installing Python environment"
if uv sync --dev; then
  echo "==> uv sync completed"
else
  echo "==> uv sync failed or is unavailable for this project; falling back to uv venv + uv pip install"
  uv venv
  uv pip install -e ".[dev]"
fi

echo "==> Running small pytest subset"
if [ -f tests/test_density_separability_factorial.py ]; then
  uv run python -m pytest tests/test_density_separability_factorial.py
else
  uv run python -m pytest tests/test_topology.py tests/test_allocation_shape.py
fi

echo ""
echo "Bootstrap complete. Next commands:"
echo "  tmux new -s density-threshold"
echo "  bash scripts/run_vultr_density_threshold.sh"
echo ""
echo "Optional environment overrides:"
echo "  SEED_SHARDS='0-4 5-9 10-14 15-19' bash scripts/run_vultr_density_threshold.sh"
echo "  SCP_TARGET=user@your-workstation:/path/to/destination bash scripts/run_vultr_density_threshold.sh"
