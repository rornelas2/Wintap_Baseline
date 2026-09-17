#!/bin/bash
#SBATCH --job-name=preproc_all
#SBATCH --output=/home/rornelas5/Wintap_Baseline/agent_evaluation/preproc_%j.log
#SBATCH --error=/home/rornelas5/Wintap_Baseline/agent_evaluation/preproc_%j.err
#SBATCH --partition=test
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00

set -euo pipefail
echo "Starting preprocessing on $(hostname) at $(date)"
python3 /home/rornelas5/Wintap_Baseline/agent_evaluation/preprocess_traces.py
echo "Completed preprocessing at $(date)"
