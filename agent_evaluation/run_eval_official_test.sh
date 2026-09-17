#!/bin/bash
#SBATCH --job-name=llm_official_test
#SBATCH --output=/home/rornelas5/Wintap_Baseline/agent_evaluation/eval_official_test_%j.log
#SBATCH --error=/home/rornelas5/Wintap_Baseline/agent_evaluation/eval_official_test_%j.err
#SBATCH --partition=short
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=03:00:00

set -euo pipefail

# Source ~/.bashrc to pick up API credentials if not already exported
if [ -f "$HOME/.bashrc" ]; then
    set +u
    source "$HOME/.bashrc"
    set -u
fi

ENV_FILE="/home/rornelas5/Wintap_Baseline/agent_evaluation/.env"
if [ -f "$ENV_FILE" ]; then
    set -a
    source "$ENV_FILE"
    set +a
fi

export WINTAP_LLM_BASE_URL="${WINTAP_LLM_BASE_URL:-https://opencode.ai/zen/go/v1}"
export WINTAP_LLM_MODEL="${WINTAP_LLM_MODEL:-glm-5.3-flash}"
export WINTAP_LLM_API_KEY="${WINTAP_LLM_API_KEY:-${OPENAI_API_KEY:-EMPTY}}"

if [ "$WINTAP_LLM_API_KEY" = "EMPTY" ] || [ -z "$WINTAP_LLM_API_KEY" ]; then
    echo "ERROR: WINTAP_LLM_API_KEY or OPENAI_API_KEY is not set!"
    exit 1
fi

echo "=========================================================="
echo "Evaluating LLM Agent on Full Official Wintap Test Set"
echo "Split: Official Test (truth_labels_test.json, N = 24,747)"
echo "Model: $WINTAP_LLM_MODEL | Base URL: $WINTAP_LLM_BASE_URL"
echo "Concurrency: 30 parallel workers"
echo "Start Time: $(date)"
echo "=========================================================="

echo "--- Step 1: Zero-Shot Evaluation on Full Official Test Set (24,747 samples) ---"
python3 -u /home/rornelas5/Wintap_Baseline/agent_evaluation/run_agent_evaluation.py \
  --config zero_shot \
  --split official_test \
  --concurrency 30 \
  --model "$WINTAP_LLM_MODEL" \
  --base-url "$WINTAP_LLM_BASE_URL"

echo "=========================================================="
echo "Official test evaluation successfully completed at $(date)"
echo "=========================================================="
