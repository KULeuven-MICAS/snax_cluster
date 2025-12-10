#!/usr/bin/env bash
set -euo pipefail

# Background launcher for regression tests
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUTPUT_ROOT="${ROOT_DIR}/regression_test_out"
mkdir -p "${OUTPUT_ROOT}"
BG_LOG="${OUTPUT_ROOT}/background_process.log"

# Run the regression tests in the background and capture any errors/output
nohup bash "${ROOT_DIR}/scripts/regression_test.sh" >"${BG_LOG}" 2>&1 &
echo "Regression tests started in background (PID $!). Logs will appear under regression_test_out/" >&2
