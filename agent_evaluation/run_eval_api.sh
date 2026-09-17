#!/bin/bash
#SBATCH --job-name=agent_eval_luna
#SBATCH --output=/home/rornelas5/Wintap_Baseline/agent_evaluation/eval_luna_%j.log
#SBATCH --error=/home/rornelas5/Wintap_Baseline/agent_evaluation/eval_luna_%j.err
#SBATCH --partition=short
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00

set -euo pipefail

ENV_FILE="/home/rornelas5/Wintap_Baseline/agent_evaluation/.env"
if [ -f "$ENV_FILE" ]; then
    echo "Loading environment from $ENV_FILE"
    set -a
    source "$ENV_FILE"
    set +a
fi

export WINTAP_LLM_BASE_URL="${WINTAP_LLM_BASE_URL:-https://api.openai.com/v1}"
export WINTAP_LLM_MODEL="${WINTAP_LLM_MODEL:-qwen3.8-flash}"
export WINTAP_LLM_API_KEY="${WINTAP_LLM_API_KEY:-${OPENAI_API_KEY:-EMPTY}}"

# Auto-correct if user exported API key into WINTAP_LLM_BASE_URL
if [[ "$WINTAP_LLM_BASE_URL" =~ ^sk- ]]; then
    echo "[Auto-Config] Detected API key in WINTAP_LLM_BASE_URL; reassigning to WINTAP_LLM_API_KEY."
    if [[ "$WINTAP_LLM_API_KEY" =~ ^https?:// ]]; then
        TMP="$WINTAP_LLM_BASE_URL"
        WINTAP_LLM_BASE_URL="$WINTAP_LLM_API_KEY"
        WINTAP_LLM_API_KEY="$TMP"
    else
        WINTAP_LLM_API_KEY="$WINTAP_LLM_BASE_URL"
        WINTAP_LLM_BASE_URL="https://api.openai.com/v1"
    fi
elif [[ "$WINTAP_LLM_API_KEY" =~ ^https?:// ]] && [[ ! "$WINTAP_LLM_BASE_URL" =~ ^https?:// ]]; then
    TMP="$WINTAP_LLM_BASE_URL"
    WINTAP_LLM_BASE_URL="$WINTAP_LLM_API_KEY"
    WINTAP_LLM_API_KEY="$TMP"
fi

if [[ "$WINTAP_LLM_BASE_URL" =~ "opencode.ai" ]] && [ "$WINTAP_LLM_MODEL" = "gpt-5.6-luna" ]; then
    echo "[Auto-Config] 'gpt-5.6-luna' returns upstream HTTP 500 on opencode.ai. Routing to 'qwen3.8-flash'."
    export WINTAP_LLM_MODEL="qwen3.8-flash"
fi

if [ "$WINTAP_LLM_API_KEY" = "EMPTY" ] || [ -z "$WINTAP_LLM_API_KEY" ]; then
    echo "ERROR: WINTAP_LLM_API_KEY or OPENAI_API_KEY is not set!"
    echo "Please export OPENAI_API_KEY or WINTAP_LLM_API_KEY in your shell environment, or configure .env"
    exit 1
fi

echo "=========================================================="
echo "Running Full Clean Temporal Agent Evaluation on Slurm"
echo "Model: $WINTAP_LLM_MODEL | Base URL: $WINTAP_LLM_BASE_URL"
echo "Concurrency: 25 workers"
echo "Start: $(date)"
echo "=========================================================="

echo "--- Step 1: Verification Pilot (20 samples) ---"
python3 -u /home/rornelas5/Wintap_Baseline/agent_evaluation/run_agent_evaluation.py \
  --config all \
  --split pilot_20 \
  --concurrency 5 \
  --model "$WINTAP_LLM_MODEL" \
  --base-url "$WINTAP_LLM_BASE_URL"

echo "--- Step 2: Full Zero-Shot Clean 2024 Evaluation (14,619 samples) ---"
python3 -u /home/rornelas5/Wintap_Baseline/agent_evaluation/run_agent_evaluation.py \
  --config zero_shot \
  --split 2024 \
  --concurrency 25 \
  --model "$WINTAP_LLM_MODEL" \
  --base-url "$WINTAP_LLM_BASE_URL"

echo "--- Step 3: Full Retrieval-Assisted Clean 2024 Evaluation (14,619 samples) ---"
python3 -u /home/rornelas5/Wintap_Baseline/agent_evaluation/run_agent_evaluation.py \
  --config retrieval_assisted \
  --split 2024 \
  --concurrency 25 \
  --model "$WINTAP_LLM_MODEL" \
  --base-url "$WINTAP_LLM_BASE_URL"

echo "=========================================================="
echo "All evaluations successfully completed at $(date)"
echo "=========================================================="
