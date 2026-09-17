#!/bin/bash
#SBATCH --job-name=test_summary
#SBATCH --output=/home/rornelas5/Wintap_Baseline/agent_evaluation/test_summary_%j.log
#SBATCH --error=/home/rornelas5/Wintap_Baseline/agent_evaluation/test_summary_%j.err
#SBATCH --partition=test
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:10:00

set -euo pipefail
python3 /home/rornelas5/Wintap_Baseline/agent_evaluation/test_summary_builder.py
