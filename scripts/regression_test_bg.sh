#!/usr/bin/env bash
set -euo pipefail

# Background launcher for regression tests
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
nohup bash "${ROOT_DIR}/scripts/regression_test_local.sh" >/dev/null 2>&1 &
echo "Regression tests started in background (PID $!). Logs will appear under regression_test_out/." >&2
