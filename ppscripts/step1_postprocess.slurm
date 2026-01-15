#!/bin/bash
#SBATCH --job-name=step1_pp
#SBATCH --output=logs/step1_pp_%j.log
#SBATCH --error=logs/step1_pp_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=04:00:00
#SBATCH --mem=12G

echo "Starting Step 1: Post-processing Data-Driven Sims on $(hostname)"
echo "Date: $(date)"

python run_process_dd.py
