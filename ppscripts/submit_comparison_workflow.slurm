#!/bin/bash
#SBATCH --job-name=comparison_workflow
#SBATCH --output=logs/comp_workflow_%j.log
#SBATCH --error=logs/comp_workflow_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=04:00:00
#SBATCH --mem=24G

# Load environment (Adapt if you use modules, e.g., module load python)
# source activate python_env 

echo "Starting Comparison Workflow on $(hostname)"
echo "Date: $(date)"

# 1. Extraction Phase
echo "----------------------------------------------------------------"
echo "PHASE 1: Extracting Simulation Data to NPZ"
echo "----------------------------------------------------------------"
python extract_all_for_comparison.py

if [ $? -ne 0 ]; then
    echo "Error during extraction. Exiting."
    exit 1
fi

# 2. Comparison Phase
echo "----------------------------------------------------------------"
echo "PHASE 2: Comparing Experiments to Simulations"
echo "----------------------------------------------------------------"

# Ensure directories exist
mkdir -p data_plots rankings

# Run the comparison
# Arguments: [experiments_csv_folder] [orig_sims_npz] [datadriven_sims_npz]
python compare_all_experiments.py \
    data_experiments/ \
    extracted_npz/original/ \
    extracted_npz/datadriven/

echo "----------------------------------------------------------------"
echo "Workflow Complete."
echo "Results in: data_plots/ and rankings/"
