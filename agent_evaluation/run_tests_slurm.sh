#!/bin/bash
#SBATCH --job-name=test_protocol
#SBATCH --output=/home/rornelas5/Wintap_Baseline/agent_evaluation/test_protocol_%j.log
#SBATCH --error=/home/rornelas5/Wintap_Baseline/agent_evaluation/test_protocol_%j.err
#SBATCH --partition=test
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:10:00

set -euo pipefail
echo "Running test_evaluation_protocol.py on $(hostname) at $(date)"
python3 -u /home/rornelas5/Wintap_Baseline/agent_evaluation/test_evaluation_protocol.py
echo "Tests finished at $(date)"
