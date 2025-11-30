#!/usr/bin/env bash
set -euo pipefail

# Background launcher for smoke tests to free the terminal immediately.
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
nohup bash "${ROOT_DIR}/scripts/run-tests.sh" >/dev/null 2>&1 &
echo "Smoke tests started in background (PID $!). Logs will appear under smoke_test_out/." >&2
