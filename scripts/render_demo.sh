#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")/.."

if [ -n "${PYTHON_BIN:-}" ]; then
  PYTHON_CMD="${PYTHON_BIN}"
elif [ -x ".venv/bin/python" ]; then
  PYTHON_CMD=".venv/bin/python"
elif command -v python >/dev/null 2>&1; then
  PYTHON_CMD="python"
elif command -v python3 >/dev/null 2>&1; then
  PYTHON_CMD="python3"
else
  echo "ERROR: python not found. Run setup.sh first, or set PYTHON_BIN." >&2
  exit 1
fi

"${PYTHON_CMD}" -m brnfs demo-assets "$@"
