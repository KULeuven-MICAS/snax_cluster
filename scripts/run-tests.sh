#!/usr/bin/env bash
set -euo pipefail

# Run SimbaCore smoke tests after each commit.
TARGET_DIR="target/snitch_cluster"
CFG_OVERRIDE="cfg/snax_simbacore_cluster.hjson"
VSIM_BIN="bin/snitch_cluster.vsim"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"
OUTPUT_ROOT="${ROOT_DIR}/smoke_test_out"
TIMESTAMP="$(date -u +%Y%m%d_%H%M%S)"
mkdir -p "${OUTPUT_ROOT}"

if command -v git >/dev/null 2>&1 && git -C "${ROOT_DIR}" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  COMMIT_HASH="$(git -C "${ROOT_DIR}" rev-parse --short HEAD 2>/dev/null || echo no-git)"
else
  COMMIT_HASH="no-git"
fi

RUN_DIR="${OUTPUT_ROOT}/${TIMESTAMP}-${COMMIT_HASH}"
mkdir -p "${RUN_DIR}"
SUMMARY_FILE="${RUN_DIR}/summary.txt"
BUILD_LOG="${RUN_DIR}/build.log"

# 1) Build process → build.log
bash "${ROOT_DIR}/scripts/build.sh" > "${BUILD_LOG}" 2>&1 || true

# 2) Define tests (name:elf-path relative to ${TARGET_DIR})
declare -a TESTS=(
  "nop:sw/apps/nop/build/nop.elf"
  "snax-simbacore:sw/apps/snax-simbacore/build/snax-simbacore-main.elf"
)

# 3) Run tests, one log per test
pushd "${TARGET_DIR}" >/dev/null

{
  echo "Timestamp (UTC): $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
  echo "Commit: ${COMMIT_HASH}"
  echo "Tests:"
} > "${SUMMARY_FILE}"

for entry in "${TESTS[@]}"; do
  name="${entry%%:*}"
  elf_rel="${entry#*:}"
  test_log="${RUN_DIR}/${name}.log"
  "${VSIM_BIN}" "${elf_rel}" > "${test_log}" 2>&1

  rc=$?
  # Interpret exit code as number of errors (0 = pass). If timeout (124), count as 9999.
  if [ "${rc}" -eq 124 ]; then
    errors=9999
  else
    errors="${rc}"
  fi

  echo "  ${name}: errors=${errors}" >> "${SUMMARY_FILE}"
done

popd >/dev/null
