#!/bin/bash
#SBATCH --job-name=step2_extract
#SBATCH --output=logs/step2_extract_%j.log
#SBATCH --error=logs/step2_extract_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=04:00:00
#SBATCH --mem=24G

echo "Starting Step 2: Extraction on $(hostname)"
echo "Date: $(date)"

python extract_disp_for_comparison.py
