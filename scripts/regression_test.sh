#!/usr/bin/env bash
set -euo pipefail

# Run all specified SNAX-level tests and collect results.

TARGET_DIR="target/snitch_cluster"
CFG_OVERRIDE="cfg/snax_simbacore_cluster.hjson"
VSIM_BIN="bin/snitch_cluster.vsim"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"
OUTPUT_ROOT="${ROOT_DIR}/regression_test_out"
TIMESTAMP="$(date -u +%Y%m%d_%H%M%S)"
mkdir -p "${OUTPUT_ROOT}"

# Get commit hash
if command -v git >/dev/null 2>&1 && git -C "${ROOT_DIR}" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  COMMIT_HASH="$(git -C "${ROOT_DIR}" rev-parse --short HEAD 2>/dev/null || echo no-git)"
  COMMIT_MSG="$(git -C "${ROOT_DIR}" log -1 --pretty=%s 2>/dev/null || echo "")"
else
  COMMIT_HASH="no-git"
  COMMIT_MSG="no-git"
fi

RUN_DIR="${OUTPUT_ROOT}/${TIMESTAMP}-${COMMIT_HASH}"
mkdir -p "${RUN_DIR}"
SUMMARY_FILE="${RUN_DIR}/summary.log"
BUILD_LOG="${RUN_DIR}/build.log"

# Build process → build.log
build_rc=0
if bash "${ROOT_DIR}/scripts/build_sim.sh" > "${BUILD_LOG}" 2>&1; then
  :
else
  build_rc=$?
fi

# Define tests (names only; ELF at sw/apps/[name]/build/[name].elf)
declare -a TESTS=(
  "nop"
  "snax-simbacore-main"
  "snax-simbacore-osgemm"
)

pushd "${TARGET_DIR}" >/dev/null

{ # Summary header
  echo "Timestamp: $(date -u '+%Y-%m-%d %H:%M:%S')"
  echo "Commit: ${COMMIT_HASH}"
  echo "Message: ${COMMIT_MSG}"
  if [ "${build_rc}" -eq 0 ]; then
    echo "Build: ✅ SUCCESS"
  else
    echo "Build: ❌ FAILED (rc=${build_rc})"
  fi
  echo
  echo "Tests:"
} > "${SUMMARY_FILE}"

for name in "${TESTS[@]}"; do
  elf_rel="sw/apps/${name}/build/${name}.elf"
  test_log="${RUN_DIR}/${name}.log"

  # Run test
  "${VSIM_BIN}" "${elf_rel}" > "${test_log}" 2>&1
  
  # Parse error count from this test's log
  rc=$?
  errors=""
  parsed_errors="$(sed -n 's/.*Finished with exit code[[:space:]]\+\([0-9]\+\).*/\1/p' "${test_log}" | tail -n1)"
  if [ -z "${parsed_errors}" ]; then
    # Fallback pattern present in some logs: "Errors: N"
    parsed_errors="$(sed -n 's/.*Errors:[[:space:]]\+\([0-9]\+\).*/\1/p' "${test_log}" | tail -n1)"
  fi
  if [ -n "${parsed_errors}" ]; then
    errors="${parsed_errors}"
  elif [ "${rc}" -eq 124 ]; then
    # If timeout (124), count as a large number.
    errors=9999
  else
    # Fall back to process return code.
    errors="${rc}"
  fi
  echo "  ${name}: errors=${errors}" >> "${SUMMARY_FILE}"
done

popd >/dev/null
