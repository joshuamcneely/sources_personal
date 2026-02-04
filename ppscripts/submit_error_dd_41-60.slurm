#!/bin/bash
#SBATCH --job-name=error_dd_41-60
#SBATCH --time=04:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/error_dd_41-60_%j.log
#SBATCH --error=logs/error_dd_41-60_%j.err

# Task 2c: Error plots for dd_exp_w_factor_study 041-060 vs experiment

cd /gpfs01/home/pmyjm22/uguca_project/source/postprocessing/ppscripts

DATA_DIR="./data"
EXP_CSV="data_experiments/FS01-043-4MPa-RP-1_Event_1_displacement.csv"

mkdir -p plots/error_plots
mkdir -p logs

echo "=================================="
echo "Task 2c: dd_exp_w_factor_study 041-060 Error Analysis"
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

echo "Analyzing simulations: $SIMS"

python calculate_slip_error_norms.py \
    --batch \
    $SIMS \
    $EXP_CSV \
    --wdir $DATA_DIR \
    --group interface \
    --output plots/error_plots/dd_study_41-60_error_results.csv \
    --plot-dir plots/error_plots/dd_study_41-60

echo ""
echo "✓ Task 2c complete"
echo "Results saved to: plots/error_plots/dd_study_41-60_error_results.csv"
