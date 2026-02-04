#!/bin/bash
#SBATCH --job-name=error_gabs
#SBATCH --time=02:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=2
#SBATCH --output=logs/error_gabs_%j.log
#SBATCH --error=logs/error_gabs_%j.err

# Task 1: Error plots for gabs_fine simulations vs experiment

cd /gpfs01/home/pmyjm22/uguca_project/source/postprocessing/ppscripts

DATA_DIR="./data"
EXP_CSV="data_experiments/FS01-043-4MPa-RP-1_Event_1_displacement.csv"

# Create output directories
mkdir -p plots/error_plots
mkdir -p logs

echo "=================================="
echo "Task 1: gabs_fine Error Analysis"
echo "=================================="

# List of gabs_fine simulations
SIMS="gabs_fine_060,gabs_fine_078,gabs_fine_096,gabs_fine_112,gabs_fine_118,gabs_fine_124,gabs_fine_136,gabs_fine_142,gabs_fine_154,gabs_fine_160"

python calculate_slip_error_norms.py \
    --batch \
    $SIMS \
    $EXP_CSV \
    --wdir $DATA_DIR \
    --group interface \
    --output plots/error_plots/gabs_fine_error_results.csv \
    --plot-dir plots/error_plots/gabs_fine

echo ""
echo "✓ Task 1 complete"
echo "Results saved to: plots/error_plots/gabs_fine_error_results.csv"
