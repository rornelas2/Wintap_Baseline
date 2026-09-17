#!/bin/bash
#SBATCH --job-name=pilot_muse_gpu
#SBATCH --output=/home/rornelas5/Wintap_Baseline/agent_evaluation/pilot_muse_gpu_%j.log
#SBATCH --error=/home/rornelas5/Wintap_Baseline/agent_evaluation/pilot_muse_gpu_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=01:00:00

set -euo pipefail

echo "=========================================================="
echo "Running Self-Contained Muse-Glimmer Pilot Evaluation on $(hostname)"
echo "Job ID: $SLURM_JOB_ID | Node: $(hostname) | Start: $(date)"
echo "=========================================================="

cd /home/rornelas5/OpenSource_Agentic_Model_Setup
export TUTORIAL_DIR="$PWD"
source scripts/activate-muse-gguf.sh
export MUSE_GPU=a100
export MUSE_PORT=8000

# Background server management
server_pid=''
cleanup() {
  if [[ -n "$server_pid" ]]; then
    echo "Stopping background Muse server (PID: $server_pid)..."
    kill "$server_pid" 2>/dev/null || true
    wait "$server_pid" 2>/dev/null || true
    echo "Muse server stopped."
  fi
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

server_log="/home/rornelas5/Wintap_Baseline/agent_evaluation/muse_server_${SLURM_JOB_ID}.log"
echo "Launching Muse-Glimmer llama-server on 127.0.0.1:$MUSE_PORT (logging to $server_log)..."
bash scripts/serve-muse-gguf.sh > "$server_log" 2>&1 &
server_pid=$!

echo "Waiting for Muse-Glimmer server to become healthy..."
ready=0
for ((i=0; i<60; i++)); do
  kill -0 "$server_pid" 2>/dev/null || {
    echo "ERROR: Server process died unexpectedly! Last log lines:"
    tail -n 40 "$server_log"
    exit 1
  }
  if curl -fsS --max-time 2 "http://127.0.0.1:${MUSE_PORT}/health" >/dev/null 2>&1; then
    ready=1
    break
  fi
  sleep 2
done

if [[ "$ready" -ne 1 ]]; then
  echo "ERROR: Server health check timed out after 120s! Last log lines:"
  tail -n 40 "$server_log"
  exit 1
fi

echo "Muse-Glimmer server is ready! Checking /v1/models:"
curl -s "http://127.0.0.1:${MUSE_PORT}/v1/models"
echo ""

echo "=========================================================="
echo "Launching Agent Evaluation (20 samples, zero-shot and retrieval-assisted)..."
echo "=========================================================="

python3 -u /home/rornelas5/Wintap_Baseline/agent_evaluation/run_agent_evaluation.py \
  --config all \
  --split pilot_20 \
  --base-url "http://127.0.0.1:${MUSE_PORT}/v1" \
  --model "muse-glimmer-30b-dynamic" \
  --timeout 120

echo "=========================================================="
echo "Pilot Evaluation Finished Successfully at: $(date)"
echo "=========================================================="
