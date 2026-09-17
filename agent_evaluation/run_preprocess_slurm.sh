#!/bin/bash
#SBATCH --job-name=preproc_trees
#SBATCH --output=/home/rornelas5/Wintap_Baseline/agent_evaluation/preproc_%A_%a.log
#SBATCH --error=/home/rornelas5/Wintap_Baseline/agent_evaluation/preproc_%A_%a.err
#SBATCH --partition=test
#SBATCH --array=0-7
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:30:00

set -euo pipefail
echo "Starting shard ${SLURM_ARRAY_TASK_ID} on $(hostname) at $(date)"
python3 /home/rornelas5/Wintap_Baseline/agent_evaluation/preprocess_traces.py --shard "${SLURM_ARRAY_TASK_ID}"
echo "Completed shard ${SLURM_ARRAY_TASK_ID} at $(date)"
