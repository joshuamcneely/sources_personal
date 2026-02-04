#!/bin/bash
#SBATCH --job-name=overlay_dd_41-60
#SBATCH --time=04:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/overlay_dd_41-60_%j.log
#SBATCH --error=logs/overlay_dd_41-60_%j.err

# Task 5c: Multi-overlay waterfall for dd_exp_w_factor_study 041-060

cd /gpfs01/home/pmyjm22/uguca_project/source/postprocessing/ppscripts

DATA_DIR="./data"
EXP_CSV="data_experiments/FS01-043-4MPa-RP-1_Event_1_displacement.csv"

mkdir -p plots/overlay
mkdir -p logs

echo "=================================="
echo "Task 5c: dd_study 041-060 Multi-Overlay Waterfall"
echo "=================================="

# Generate simulation list 041-060
SIMS=""
for i in $(seq -f "%03g" 41 60); do
    if [ -z "$SIMS" ]; then
        SIMS="dd_exp_w_factor_study_$i"
    else
        SIMS="$SIMS,dd_exp_w_factor_study_$i"
    fi
done

echo "Overlaying simulations: $SIMS"

python waterfall_and_space_time_plot.py \
    --multi \
    interface \
    auto \
    --sims $SIMS \
    --exps $EXP_CSV

# Move to dedicated folder
mv plots/multi_overlay_*.png plots/overlay/dd_study_41-60_multi_overlay.png 2>/dev/null

echo ""
echo "✓ Task 5c complete"
echo "Plot saved to: plots/overlay/dd_study_41-60_multi_overlay.png"
