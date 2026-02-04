#!/bin/bash
#SBATCH --job-name=plot_no_nuc
#SBATCH --time=02:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=2
#SBATCH --output=logs/plot_no_nuc_%j.log
#SBATCH --error=logs/plot_no_nuc_%j.err

# Task 3: Plots for dd_no_nucleation_data_only

cd /gpfs01/home/pmyjm22/uguca_project/source/postprocessing/ppscripts

DATA_DIR="./data"
EXP_CSV="data_experiments/FS01-043-4MPa-RP-1_Event_1_displacement.csv"

mkdir -p plots/no_nucleation
mkdir -p logs

echo "=================================="
echo "Task 3: dd_no_nucleation_data_only Plots"
echo "=================================="

SIM_NAME="_dd_no_nucleation_data_only"

# Waterfall + Heatmap plots
echo "Generating waterfall and heatmap plots..."
python waterfall_and_space_time_plot.py \
    $SIM_NAME \
    interface \
    auto \
    $EXP_CSV

# Move plots to dedicated folder
mv plots/${SIM_NAME}_*.png plots/no_nucleation/ 2>/dev/null

# Error analysis
echo ""
echo "Calculating error norms..."
python calculate_slip_error_norms.py \
    $SIM_NAME \
    $EXP_CSV \
    --wdir $DATA_DIR \
    --group interface

echo ""
echo "✓ Task 3 complete"
echo "Plots saved to: plots/no_nucleation/"
