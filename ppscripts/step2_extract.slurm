#!/bin/bash
#SBATCH --job-name=step2_extract
#SBATCH --output=logs/step2_extract_%A_%a.log
#SBATCH --error=logs/step2_extract_%A_%a.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=04:00:00
#SBATCH --mem=12G
#SBATCH --array=1-14

echo "Starting Step 2: Extraction Task $SLURM_ARRAY_TASK_ID on $(hostname)"
echo "Date: $(date)"

python -u run_extract_disp.py $SLURM_ARRAY_TASK_ID
