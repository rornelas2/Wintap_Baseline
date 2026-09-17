#!/bin/bash
#SBATCH --job-name=pilot_mock
#SBATCH --output=/home/rornelas5/Wintap_Baseline/agent_evaluation/pilot_mock_%j.log
#SBATCH --error=/home/rornelas5/Wintap_Baseline/agent_evaluation/pilot_mock_%j.err
#SBATCH --partition=test
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:05:00

set -euo pipefail
echo "Running pilot mock evaluation on $(hostname) at $(date)"
python3 -u /home/rornelas5/Wintap_Baseline/agent_evaluation/run_agent_evaluation.py \
  --config all \
  --split pilot_20 \
  --mock
echo "Pilot mock finished at $(date)"
