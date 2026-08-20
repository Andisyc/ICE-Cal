#!/bin/zsh
set -euo pipefail

ATLAS_ROOT="${0:A:h}"
PORT="${PORT:-8767}"
BASE_URL="http://127.0.0.1:${PORT}"
TARGET_PATH="${ATLAS_PAGE:-${1:-/index.html}}"
LOG_PATH="${ATLAS_LOG_PATH:-/tmp/ice-cal-femr-context-atlas-${PORT}.log}"
PID_PATH="${ATLAS_PID_PATH:-/tmp/ice-cal-femr-context-atlas-${PORT}.pid}"

atlas_is_ready() {
  /usr/bin/curl -fsS "${BASE_URL}/healthz" 2>/dev/null \
    | /usr/bin/grep -q '"service":"ice-cal-femr-context-atlas"'
}

if atlas_is_ready; then
  print "[Atlas] reuse: ${BASE_URL}${TARGET_PATH}"
  if [[ "$(/usr/bin/uname -s)" == "Darwin" ]]; then
    /usr/bin/open "${BASE_URL}${TARGET_PATH}"
  else
    print "${BASE_URL}${TARGET_PATH}"
  fi
  exit 0
fi

if [[ ! -f "${ATLAS_ROOT}/auxiliary/atlas_app/node_modules/roughjs/bundled/rough.esm.js" ]]; then
  print -u2 "[Atlas] missing roughjs dependency"
  print -u2 "[Atlas] run: npm --prefix '${ATLAS_ROOT}/auxiliary/atlas_app' install"
  exit 1
fi

print "[Atlas] starting background server: ${BASE_URL}"
print "[Atlas] log: ${LOG_PATH}"

nohup /usr/bin/env PORT="${PORT}" node "${ATLAS_ROOT}/auxiliary/atlas_app/serve_architecture.mjs" \
  >"${LOG_PATH}" 2>&1 </dev/null &
SERVER_PID="$!"
print "${SERVER_PID}" >"${PID_PATH}"

for _attempt in {1..50}; do
  if atlas_is_ready; then
    print "[Atlas] ready: ${BASE_URL}${TARGET_PATH}"
    if [[ "$(/usr/bin/uname -s)" == "Darwin" ]]; then
      /usr/bin/open "${BASE_URL}${TARGET_PATH}"
    else
      print "${BASE_URL}${TARGET_PATH}"
    fi
    exit 0
  fi
  /bin/sleep 0.1
done

print -u2 "[Atlas] server failed to become ready at ${BASE_URL}"
print -u2 "[Atlas] inspect log: ${LOG_PATH}"
exit 1
