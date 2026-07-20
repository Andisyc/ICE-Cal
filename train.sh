#!/usr/bin/env bash

# Ordinary DAgger training launcher.  This is deliberately separate from
# start.sh (interactive playback) and formal Gate 0 materialization.
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"

WORKFLOW_MODE=""
RUN_NAME=""
RESUME_RUN=""
ARTIFACT_DIR_INPUT=""
TASK="g1_walk_flat"
SIM="mujoco"
WORKFLOW="g1_walk_stand"
DEVICE="cuda:0"
EXECUTION_MODE=""
DRY_RUN=0
HYDRA_OVERRIDES=()

usage() {
    cat <<'EOF'
Usage:
  ./train.sh --workflow-mode fresh [options] [hydra.override=value ...]
  ./train.sh --workflow-mode resume --resume-run <run_dir> [options] [hydra.override=value ...]

Modes:
  fresh   Create one paired, time-sorted run/artifact identity.
  resume  Continue only the explicit run containing run_manifest.json.

Options:
  --run-name <name>            Fresh name suffix (default: g1_walk_stand_distill).
  --resume-run <dir>           Existing workflow run; required for resume.
  --artifact-dir <dir>         Existing role-artifact root for an unpaired resume.
  --task <task>                Distill CLI task (default: g1_walk_flat).
  --sim <backend>              Distill CLI backend (default: mujoco).
  --workflow <profile>         Workflow Hydra profile (default: g1_walk_stand).
  --device <device>            Training device (default: cuda:0).
  --execution-mode <mode>      legacy or persistent_async; omitted keeps config default.
  --dry-run                    Print the exact uv command without training.
  -h, --help                   Show this help.

Examples:
  ./train.sh --workflow-mode fresh --run-name walk-stop-r3 \
    --execution-mode persistent_async training.workflow.dagger_iterations=8

  ./train.sh --workflow-mode resume \
    --resume-run logs/distill_workflow/20260720-140520_walk-stop-r3
EOF
}

die() {
    printf 'Usage error: %s\n' "$*" >&2
    exit 2
}

require_value() {
    local flag="$1"
    local value="${2:-}"
    if [ -z "$value" ]; then
        die "$flag requires a value"
    fi
}

resolve_existing_dir() {
    local value="$1"
    local candidate
    if [[ "$value" = /* ]]; then
        candidate="$value"
    else
        candidate="$ROOT_DIR/$value"
    fi
    [ -d "$candidate" ] || die "directory does not exist: $candidate"
    (cd -- "$candidate" && pwd -P)
}

while [ "$#" -gt 0 ]; do
    case "$1" in
        -h|--help)
            usage
            exit 0
            ;;
        --dry-run)
            DRY_RUN=1
            shift
            ;;
        --workflow-mode)
            require_value "$1" "${2:-}"
            WORKFLOW_MODE="$2"
            shift 2
            ;;
        --run-name)
            require_value "$1" "${2:-}"
            RUN_NAME="$2"
            shift 2
            ;;
        --resume-run)
            require_value "$1" "${2:-}"
            RESUME_RUN="$2"
            shift 2
            ;;
        --artifact-dir)
            require_value "$1" "${2:-}"
            ARTIFACT_DIR_INPUT="$2"
            shift 2
            ;;
        --task)
            require_value "$1" "${2:-}"
            TASK="$2"
            shift 2
            ;;
        --sim)
            require_value "$1" "${2:-}"
            SIM="$2"
            shift 2
            ;;
        --workflow)
            require_value "$1" "${2:-}"
            WORKFLOW="$2"
            shift 2
            ;;
        --device)
            require_value "$1" "${2:-}"
            DEVICE="$2"
            shift 2
            ;;
        --execution-mode)
            require_value "$1" "${2:-}"
            EXECUTION_MODE="$2"
            shift 2
            ;;
        --)
            shift
            HYDRA_OVERRIDES+=("$@")
            break
            ;;
        *=*)
            HYDRA_OVERRIDES+=("$1")
            shift
            ;;
        *)
            die "unrecognized argument: $1"
            ;;
    esac
done

case "$WORKFLOW_MODE" in
    fresh|resume) ;;
    "") die "--workflow-mode fresh or --workflow-mode resume is required" ;;
    *) die "--workflow-mode must be fresh or resume, got $WORKFLOW_MODE" ;;
esac

if [ -n "$EXECUTION_MODE" ] && [ "$EXECUTION_MODE" != "legacy" ] && [ "$EXECUTION_MODE" != "persistent_async" ]; then
    die "--execution-mode must be legacy or persistent_async"
fi

case "${UNILAB_NATIVE_HEAP_DEBUG:-0}" in
    0) NATIVE_HEAP_DEBUG=0 ;;
    1)
        NATIVE_HEAP_DEBUG=1
        export PYTHONMALLOC="${PYTHONMALLOC:-debug}"
        export PYTHONFAULTHANDLER="${PYTHONFAULTHANDLER:-1}"
        export MALLOC_CHECK_="${MALLOC_CHECK_:-3}"
        export MALLOC_PERTURB_="${MALLOC_PERTURB_:-165}"
        ;;
    *) die "UNILAB_NATIVE_HEAP_DEBUG must be 0 or 1" ;;
esac

case "${UNILAB_NATIVE_ABORT_ON_CORRUPTION:-0}" in
    0) NATIVE_ABORT_ON_CORRUPTION=0 ;;
    1) NATIVE_ABORT_ON_CORRUPTION=1 ;;
    *) die "UNILAB_NATIVE_ABORT_ON_CORRUPTION must be 0 or 1" ;;
esac

for override in "${HYDRA_OVERRIDES[@]-}"; do
    [ -n "$override" ] || continue
    case "$override" in
        workflow=*|+workflow=*|++workflow=*|\
        training.workflow.enabled=*|+training.workflow.enabled=*|++training.workflow.enabled=*|\
        training.workflow.mode=*|+training.workflow.mode=*|++training.workflow.mode=*|\
        training.workflow.run_dir=*|+training.workflow.run_dir=*|++training.workflow.run_dir=*|\
        training.workflow.artifact_dir=*|+training.workflow.artifact_dir=*|++training.workflow.artifact_dir=*)
            die "route-defining Hydra override is owned by train.sh: $override"
            ;;
    esac
done

if [ "$WORKFLOW_MODE" = "fresh" ]; then
    [ -z "$RESUME_RUN" ] || die "--resume-run is only valid with --workflow-mode resume"
    [ -z "$ARTIFACT_DIR_INPUT" ] || die "--artifact-dir is only valid with --workflow-mode resume"
    RUN_NAME="${RUN_NAME:-g1_walk_stand_distill}"
    if ! [[ "$RUN_NAME" =~ ^[a-z0-9][a-z0-9_-]*$ ]]; then
        die "--run-name must start with lowercase letter/digit and use only lowercase letters, digits, underscores, or hyphens"
    fi
    STEM="$(date +%Y%m%d-%H%M%S)_${RUN_NAME}"
    RUN_DIR="$ROOT_DIR/logs/distill_workflow/$STEM"
    ARTIFACT_DIR="$ROOT_DIR/logs/distill_role_artifacts/$STEM"
    [ ! -e "$RUN_DIR" ] || die "fresh run_dir already exists: $RUN_DIR"
    [ ! -e "$ARTIFACT_DIR" ] || die "fresh artifact_dir already exists: $ARTIFACT_DIR"
else
    [ -z "$RUN_NAME" ] || die "--run-name is only valid with --workflow-mode fresh"
    [ -n "$RESUME_RUN" ] || die "--resume-run is required with --workflow-mode resume"
    RUN_DIR="$(resolve_existing_dir "$RESUME_RUN")"
    [ -f "$RUN_DIR/run_manifest.json" ] || die "resume run requires $RUN_DIR/run_manifest.json"

    if [ -n "$ARTIFACT_DIR_INPUT" ]; then
        ARTIFACT_DIR="$(resolve_existing_dir "$ARTIFACT_DIR_INPUT")"
    else
        RUN_PARENT="$(dirname -- "$RUN_DIR")"
        LOG_ROOT="$(dirname -- "$RUN_PARENT")"
        if [ "$(basename -- "$RUN_PARENT")" != "distill_workflow" ] || [ "$(basename -- "$LOG_ROOT")" != "logs" ]; then
            die "resume outside a logs/distill_workflow/<run> identity requires --artifact-dir"
        fi
        STEM="${RUN_DIR##*/}"
        ARTIFACT_DIR="$LOG_ROOT/distill_role_artifacts/$STEM"
        [ -d "$ARTIFACT_DIR" ] || die "paired resume artifact_dir does not exist: $ARTIFACT_DIR; pass --artifact-dir explicitly"
        ARTIFACT_DIR="$(resolve_existing_dir "$ARTIFACT_DIR")"
    fi
fi

COMMAND=(
    uv run train
    --algo distill
    --task "$TASK"
    --sim "$SIM"
    "workflow=$WORKFLOW"
    "training.workflow.mode=$WORKFLOW_MODE"
    "training.workflow.run_dir=$RUN_DIR"
    "training.workflow.artifact_dir=$ARTIFACT_DIR"
    "training.device=$DEVICE"
)
if [ -n "$EXECUTION_MODE" ]; then
    COMMAND+=("training.workflow.execution_mode=$EXECUTION_MODE")
fi
for override in "${HYDRA_OVERRIDES[@]-}"; do
    [ -n "$override" ] || continue
    COMMAND+=("$override")
done

printf '[train.sh] workflow_mode=%s\n' "$WORKFLOW_MODE"
printf '[train.sh] workflow_enabled=owner-cli\n'
if [ "$NATIVE_HEAP_DEBUG" = "1" ]; then
    printf '[train.sh] native_heap_debug=enabled\n'
    printf '[train.sh] allocator_env=PYTHONMALLOC=%s PYTHONFAULTHANDLER=%s MALLOC_CHECK_=%s MALLOC_PERTURB_=%s\n' \
        "$PYTHONMALLOC" "$PYTHONFAULTHANDLER" "$MALLOC_CHECK_" "$MALLOC_PERTURB_"
else
    printf '[train.sh] native_heap_debug=disabled\n'
fi
if [ "$NATIVE_ABORT_ON_CORRUPTION" = "1" ]; then
    printf '[train.sh] native_abort_on_corruption=enabled\n'
else
    printf '[train.sh] native_abort_on_corruption=disabled\n'
fi
printf '[train.sh] run_dir=%s\n' "$RUN_DIR"
printf '[train.sh] artifact_dir=%s\n' "$ARTIFACT_DIR"
if [ -n "$EXECUTION_MODE" ]; then
    printf '[train.sh] execution_mode=%s\n' "$EXECUTION_MODE"
else
    printf '[train.sh] execution_mode=config-default\n'
fi
printf '[train.sh] command:'
printf ' %q' "${COMMAND[@]}"
printf '\n'

if [ "$DRY_RUN" = "1" ]; then
    exit 0
fi

exec "${COMMAND[@]}"
