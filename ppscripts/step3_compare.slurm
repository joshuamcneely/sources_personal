#!/bin/bash
#SBATCH --job-name=step3_compare
#SBATCH --output=logs/step3_compare_%A_%a.log
#SBATCH --error=logs/step3_compare_%A_%a.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=01:00:00
#SBATCH --mem=8G
#SBATCH --array=1-10

echo "Starting Step 3: Comparison Task $SLURM_ARRAY_TASK_ID on $(hostname)"
echo "Date: $(date)"

# Usage: python compare_all_experiments.py [exp_folder] [sim_folder] [dd_sim_folder] [task_id]
python compare_all_experiments.py data_experiments extracted_npz/original extracted_npz/datadriven $SLURM_ARRAY_TASK_ID
