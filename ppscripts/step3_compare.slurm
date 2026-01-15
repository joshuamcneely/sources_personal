#!/bin/bash
#SBATCH --job-name=step3_compare
#SBATCH --output=logs/step3_compare_%j.log
#SBATCH --error=logs/step3_compare_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=01:00:00
#SBATCH --mem=8G

echo "Starting Step 3: Comparison and Plotting on $(hostname)"
echo "Date: $(date)"

# Usage: python compare_all_experiments.py [exp_folder] [sim_folder] [dd_sim_folder]
python compare_all_experiments.py data_experiments extracted_npz/original extracted_npz/datadriven
