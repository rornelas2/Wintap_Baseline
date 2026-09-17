#!/bin/bash
#SBATCH --job-name=muse_server
#SBATCH --output=/home/rornelas5/Wintap_Baseline/agent_evaluation/muse_server_%j.log
#SBATCH --error=/home/rornelas5/Wintap_Baseline/agent_evaluation/muse_server_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=04:00:00

set -euo pipefail

echo "=========================================================="
echo "Launching Muse-Glimmer Local Server on $(hostname)"
echo "Job ID: $SLURM_JOB_ID | Start: $(date)"
echo "=========================================================="

cd /home/rornelas5/OpenSource_Agentic_Model_Setup
export TUTORIAL_DIR="$PWD"

# Check GPU
nvidia-smi --query-gpu=name,memory.total --format=csv

# If running quantized Muse on A100:
source scripts/activate-muse-gguf.sh
export MUSE_GPU=a100
export MUSE_PORT=8000
echo "Starting llama-server for Muse-Glimmer on port $MUSE_PORT..."
bash scripts/serve-muse-gguf.sh
