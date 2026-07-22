#!/usr/bin/env bash
set -euo pipefail

CRASH="${1:-/var/crash/_usr_bin_python3.10.1005.crash}"
WORK_ROOT="${WORK_ROOT:-/ssd1/cyx/UniLab/logs/native_core}"
STAMP="$(date +%Y%m%d-%H%M%S)"
OUT="${OUT:-${WORK_ROOT}/${STAMP}_python310_apport_core}"

if [[ ! -f "$CRASH" ]]; then
  echo "missing crash report: $CRASH" >&2
  exit 2
fi

if ! command -v apport-unpack >/dev/null 2>&1; then
  echo "missing apport-unpack; install with: sudo apt-get install -y apport" >&2
  exit 2
fi

if ! command -v gdb >/dev/null 2>&1; then
  echo "missing gdb; install with: sudo apt-get install -y gdb python3.10-dbg libc6-dbg" >&2
  exit 2
fi

mkdir -p "$OUT"

apport-unpack "$CRASH" "$OUT/apport-unpacked" >"$OUT/apport-unpack.log" 2>&1

EXE="$(cat "$OUT/apport-unpacked/ExecutablePath")"
CORE="$OUT/apport-unpacked/CoreDump"
GDB_CMDS="$OUT/gdb-python-commands.txt"
GDB_OUT="$OUT/gdb-python-stack.txt"
HELPER="${PYTHON_GDB_HELPER:-}"

if [[ -z "$HELPER" ]]; then
  for candidate in \
    /usr/share/gdb/auto-load/usr/bin/python3.10-gdb.py \
    /usr/share/gdb/auto-load/usr/bin/python3.10m-gdb.py \
    /usr/share/gdb/auto-load/ssd1/cyx/UniLab/.venv/bin/python3-gdb.py; do
    if [[ -f "$candidate" ]]; then
      HELPER="$candidate"
      break
    fi
  done
fi

{
  echo "set pagination off"
  echo "set print pretty on"
  echo "set auto-load safe-path /"
  if [[ -n "$HELPER" ]]; then
    echo "source $HELPER"
  fi
  echo "info auto-load python-scripts"
  echo "info threads"
  echo "thread apply all bt 30"
  echo "thread apply all py-bt"
  echo "thread apply all py-list"
  echo "thread 1"
  echo "bt full"
  echo "py-bt-full"
  echo "py-list"
  echo "info sharedlibrary"
  echo "quit"
} >"$GDB_CMDS"

set +e
gdb -q "$EXE" "$CORE" -batch -x "$GDB_CMDS" >"$GDB_OUT" 2>&1
GDB_RC=$?
set -e

{
  echo "crash=$CRASH"
  echo "out=$OUT"
  echo "exe=$EXE"
  echo "core=$CORE"
  echo "helper=${HELPER:-not-found}"
  echo "gdb_returncode=$GDB_RC"
  echo "proc_cmdline:"
  cat "$OUT/apport-unpacked/ProcCmdline" 2>/dev/null || true
  echo
  echo "signal:"
  cat "$OUT/apport-unpacked/Signal" 2>/dev/null || true
  echo
  echo "python_gdb_status:"
  if grep -q 'Undefined command: "py-bt"' "$GDB_OUT"; then
    echo "py-bt-unavailable; install python3.10-dbg or set PYTHON_GDB_HELPER=/path/to/python-gdb.py"
  else
    echo "py-bt-attempted; inspect gdb-python-stack.txt"
  fi
} >"$OUT/SUMMARY.txt"

ARCHIVE="${OUT}-RETURN_ME.tar.gz"
tar -czf "$ARCHIVE" -C "$(dirname "$OUT")" "$(basename "$OUT")"

cat "$OUT/SUMMARY.txt"
echo "archive=$ARCHIVE"
