#!/bin/bash

# 获取当前操作系统名称
OS="$(uname -s)"

ALGO="sac"
TASK="g1_walk_flat"
SIM="mujoco"
DRY_RUN=0
POSITIONAL_ARGS=()

while [ "$#" -gt 0 ]; do
    case "$1" in
        --dry-run)
            DRY_RUN=1
            shift
            ;;
        --algo)
            if [ "$#" -lt 2 ]; then
                echo "Usage error: --algo requires a value"
                exit 2
            fi
            ALGO="$2"
            shift 2
            ;;
        --task)
            if [ "$#" -lt 2 ]; then
                echo "Usage error: --task requires a value"
                exit 2
            fi
            TASK="$2"
            shift 2
            ;;
        --sim)
            if [ "$#" -lt 2 ]; then
                echo "Usage error: --sim requires a value"
                exit 2
            fi
            SIM="$2"
            shift 2
            ;;
        g1_walk_flat|g1_stand_still|g1_walk_height)
            TASK="$1"
            shift
            ;;
        *)
            POSITIONAL_ARGS+=("$1")
            shift
            ;;
    esac
done

KEYBOARD="true"
if [ "$TASK" = "g1_stand_still" ]; then
    KEYBOARD="false"
fi

if [ -z "${UNILAB_G1_ACTION_TRACE_INTERVAL+x}" ]; then
    export UNILAB_G1_ACTION_TRACE_INTERVAL=20
fi

# 将那些两边都一样的长参数提取出来，方便以后修改
ARGS=(
    scripts/play_interactive.py
    --algo "$ALGO"
    --task "$TASK"
    --sim "$SIM"
    interactive.action_mode=policy
    "interactive.keyboard=${KEYBOARD}"
)

# 便捷用法：
#   ./start.sh 2026-06-12_15-46-01_mujoco
#   ./start.sh --task g1_stand_still 2026-07-09_00-00-00_mujoco
# 等价于追加：
#   algo.load_run=<RUN_NAME> +algo.checkpoint=model_5000.pt
if [ "${#POSITIONAL_ARGS[@]}" -gt 0 ] && [[ "${POSITIONAL_ARGS[0]}" != *=* ]] && [[ "${POSITIONAL_ARGS[0]}" != -* ]]; then
    RUN_NAME="${POSITIONAL_ARGS[0]}"
    POSITIONAL_ARGS=("${POSITIONAL_ARGS[@]:1}")
    ARGS+=(
        "algo.load_run=${RUN_NAME}"
        "+algo.checkpoint=model_5000.pt"
    )
fi

ARGS+=("${POSITIONAL_ARGS[@]}")

echo "[start.sh] task=${TASK} sim=${SIM} algo=${ALGO} keyboard=${KEYBOARD}"
if [ -n "${UNILAB_G1_ACTION_TRACE+x}" ]; then
    echo "[start.sh] action_trace=${UNILAB_G1_ACTION_TRACE} interval=${UNILAB_G1_ACTION_TRACE_INTERVAL}"
fi
printf '[start.sh] command:'
printf ' %q' "${ARGS[@]}"
printf '\n'

if [ "$DRY_RUN" = "1" ]; then
    exit 0
fi

# 根据操作系统执行不同的命令
if [ "$OS" = "Darwin" ]; then
    # Mac OS 系统 (Darwin) 必须使用 mjpython
    echo "🍏 检测到 macOS，正在使用 mjpython 启动仿真..."
    uv run mjpython "${ARGS[@]}"
elif [ "$OS" = "Linux" ]; then
    # 4090 服务器 (Linux) 直接使用默认的 python 即可
    echo "🐧 检测到 Linux，正在使用常规 python 启动仿真..."
    uv run python "${ARGS[@]}"
else
    echo "⚠️ 未知的操作系统: $OS"
    exit 1
fi
