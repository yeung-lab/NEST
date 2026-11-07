#!/bin/bash
# Run NEST evaluation. Reproducibility constants below mirror the
# configuration used for our reported numbers.
#
# Usage:
#   bash nest_eval/run.sh
#   MODEL=vllm/Qwen/Qwen3-VL-4B-Instruct bash nest_eval/run.sh
#   MODEL=... JUDGE_MODEL="$JUDGE_MODEL_LARGE" EXTRA_FLAGS="--limit 20" bash nest_eval/run.sh
#
# Wrap with sbatch yourself if running under SLURM.

# ---- Reproducibility constants ----
MAX_MODEL_LEN=32768
MAX_TOKENS=256
GPU_MEM=0.7
JUDGE_GPU_MEM=0.7
JUDGE_MAX_MODEL_LEN=4096
TEMPERATURE=0.1
BATCH_SIZE=1
PORT=9000
JUDGE_PORT=9001

# ---- Judge model presets ----
JUDGE_MODEL_TEST="vllm/Qwen/Qwen3-VL-4B-Instruct"
JUDGE_MODEL_LARGE="vllm/Qwen/Qwen3-VL-32B-Instruct"   # used for final reported results
export JUDGE_MODEL_TEST JUDGE_MODEL_LARGE

# ---- User-overridable ----
MODEL="${MODEL:-vllm/Qwen/Qwen3-VL-8B-Instruct}"
JUDGE_MODEL="${JUDGE_MODEL:-$JUDGE_MODEL_TEST}"
EXTRA_FLAGS="${EXTRA_FLAGS:-}"   # e.g. "--limit 20" or "--rebuild"

python -m nest_eval.run \
    --model "$MODEL" \
    --judge-model "$JUDGE_MODEL" \
    --max-model-len "$MAX_MODEL_LEN" \
    --max-tokens "$MAX_TOKENS" \
    --gpu-memory-utilization "$GPU_MEM" \
    --judge-gpu-memory-utilization "$JUDGE_GPU_MEM" \
    --judge-max-model-len "$JUDGE_MAX_MODEL_LEN" \
    --temperature "$TEMPERATURE" \
    --batch-size "$BATCH_SIZE" \
    --port "$PORT" \
    --judge-port "$JUDGE_PORT" \
    $EXTRA_FLAGS
