#!/bin/bash
#SBATCH --job-name=cmp_vs_exp
#SBATCH --output=logs/compare_vs_exp_%j.out
#SBATCH --error=logs/compare_vs_exp_%j.err
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G

# Ensure we are in the ppscripts directory
cd /gpfs01/home/pmyjm22/uguca_project/source/postprocessing/ppscripts

# Activate environment if needed
# source /path/to/venv/bin/activate

# Run comparison plot script
python3 compare_sims_vs_experiment.py
