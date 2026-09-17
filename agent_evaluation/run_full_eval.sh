#!/bin/bash
#SBATCH --job-name=full_agent_eval
#SBATCH --output=/home/rornelas5/Wintap_Baseline/agent_evaluation/full_eval_%j.log
#SBATCH --error=/home/rornelas5/Wintap_Baseline/agent_evaluation/full_eval_%j.err
#SBATCH --partition=test
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00

set -euo pipefail

ENV_FILE="/home/rornelas5/Wintap_Baseline/agent_evaluation/.env"
if [ -f "$ENV_FILE" ]; then
    set -a
    source "$ENV_FILE"
    set +a
fi

export WINTAP_LLM_BASE_URL="${WINTAP_LLM_BASE_URL:-https://api.openai.com/v1}"
export WINTAP_LLM_API_KEY="${WINTAP_LLM_API_KEY:-${OPENAI_API_KEY:-EMPTY}}"
export WINTAP_LLM_MODEL="${WINTAP_LLM_MODEL:-gpt-5.6-luna}"
CONCURRENCY="${WINTAP_CONCURRENCY:-25}"

echo "=========================================================="
echo "Running Full Clean Temporal Agent Evaluation (14,619 samples, year==2024)"
echo "Target Endpoint: $WINTAP_LLM_BASE_URL | Model: $WINTAP_LLM_MODEL"
echo "Concurrency: $CONCURRENCY workers"
echo "Start: $(date)"
echo "=========================================================="

python3 -u /home/rornelas5/Wintap_Baseline/agent_evaluation/run_agent_evaluation.py \
  --config zero_shot \
  --split 2024 \
  --concurrency "$CONCURRENCY" \
  --base-url "$WINTAP_LLM_BASE_URL" \
  --api-key "$WINTAP_LLM_API_KEY" \
  --model "$WINTAP_LLM_MODEL"

echo "Full Evaluation Finished at: $(date)"
