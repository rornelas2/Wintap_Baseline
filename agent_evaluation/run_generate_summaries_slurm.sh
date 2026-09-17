#!/bin/bash
#SBATCH --job-name=gen_summaries
#SBATCH --output=/home/rornelas5/Wintap_Baseline/agent_evaluation/gen_summaries_%j.log
#SBATCH --error=/home/rornelas5/Wintap_Baseline/agent_evaluation/gen_summaries_%j.err
#SBATCH --partition=test
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00

set -euo pipefail
echo "Running generate_all_summaries.py on $(hostname) at $(date)"
python3 -u /home/rornelas5/Wintap_Baseline/agent_evaluation/generate_all_summaries.py
echo "Completed at $(date)"
