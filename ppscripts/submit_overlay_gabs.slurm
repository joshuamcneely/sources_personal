#!/bin/bash
#SBATCH --job-name=overlay_gabs
#SBATCH --time=03:00:00
#SBATCH --mem=12G
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/overlay_gabs_%j.log
#SBATCH --error=logs/overlay_gabs_%j.err

# Task 4: Multi-overlay waterfall for gabs_fine simulations

cd /gpfs01/home/pmyjm22/uguca_project/source/postprocessing/ppscripts

DATA_DIR="./data"
EXP_CSV="data_experiments/FS01-043-4MPa-RP-1_Event_1_displacement.csv"

mkdir -p plots/overlay
mkdir -p logs

echo "=================================="
echo "Task 4: gabs_fine Multi-Overlay Waterfall"
echo "=================================="

# List of gabs_fine simulations
SIMS="gabs_fine_060,gabs_fine_078,gabs_fine_096,gabs_fine_112,gabs_fine_118,gabs_fine_124,gabs_fine_136,gabs_fine_142,gabs_fine_154,gabs_fine_160"

python waterfall_and_space_time_plot.py \
    --multi \
    interface \
    auto \
    --sims $SIMS \
    --exps $EXP_CSV

# Move to dedicated folder
mv plots/multi_overlay_*.png plots/overlay/gabs_fine_multi_overlay.png 2>/dev/null

echo ""
echo "✓ Task 4 complete"
echo "Plot saved to: plots/overlay/gabs_fine_multi_overlay.png"
