#!/bin/bash
#SBATCH --job-name=rf_reproduce
#SBATCH --output=/home/rornelas5/Wintap_Baseline/agent_evaluation/rf_reproduce_%j.log
#SBATCH --error=/home/rornelas5/Wintap_Baseline/agent_evaluation/rf_reproduce_%j.err
#SBATCH --partition=test
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:15:00

set -euo pipefail

echo "Running on node: $(hostname)"
echo "Start time: $(date)"

python3 /home/rornelas5/Wintap_Baseline/agent_evaluation/reproduce_rf.py

echo "Finished at: $(date)"
