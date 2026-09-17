#!/bin/bash
#SBATCH --job-name=inspect_trees
#SBATCH --output=/home/rornelas5/Wintap_Baseline/agent_evaluation/inspect_trees_%j.log
#SBATCH --error=/home/rornelas5/Wintap_Baseline/agent_evaluation/inspect_trees_%j.err
#SBATCH --partition=test
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:10:00

set -euo pipefail
echo "Running inspect_raw_trees.py on $(hostname)"
python3 /home/rornelas5/Wintap_Baseline/agent_evaluation/inspect_raw_trees.py
