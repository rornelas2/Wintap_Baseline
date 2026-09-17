#!/bin/bash
#SBATCH --job-name=test_speed
#SBATCH --output=/home/rornelas5/Wintap_Baseline/agent_evaluation/speed_%j.log
#SBATCH --error=/home/rornelas5/Wintap_Baseline/agent_evaluation/speed_%j.err
#SBATCH --partition=test
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:05:00

set -euo pipefail
python3 /home/rornelas5/Wintap_Baseline/agent_evaluation/test_speed.py
