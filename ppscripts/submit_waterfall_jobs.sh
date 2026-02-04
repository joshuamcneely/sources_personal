#!/bin/bash
# Submit only waterfall/overlay plotting jobs

echo "=================================="
echo "SUBMITTING WATERFALL/OVERLAY JOBS"
echo "=================================="

# Create directories
mkdir -p logs
mkdir -p plots/overlay
mkdir -p plots/no_nucleation

echo ""
echo "Submitting waterfall/overlay jobs in sequence..."
echo ""

# Task 1: dd_no_nucleation_data_only plots
echo "1. Submitting dd_no_nucleation_data_only plots..."
JOB1=$(sbatch submit_plot_no_nucleation.slurm | awk '{print $4}')
echo "   Job ID: $JOB1"

# Task 2: gabs_fine overlay
echo "2. Submitting gabs_fine multi-overlay..."
JOB2=$(sbatch submit_overlay_gabs.slurm | awk '{print $4}')
echo "   Job ID: $JOB2"

# Task 3: dd_exp_w_factor_study overlays (3 batches)
echo "3. Submitting dd_study multi-overlay (001-020)..."
JOB3=$(sbatch submit_overlay_dd_01-20.slurm | awk '{print $4}')
echo "   Job ID: $JOB3"

echo "4. Submitting dd_study multi-overlay (021-040)..."
JOB4=$(sbatch submit_overlay_dd_21-40.slurm | awk '{print $4}')
echo "   Job ID: $JOB4"

echo "5. Submitting dd_study multi-overlay (041-060)..."
JOB5=$(sbatch submit_overlay_dd_41-60.slurm | awk '{print $4}')
echo "   Job ID: $JOB5"

echo ""
echo "=================================="
echo "WATERFALL/OVERLAY JOBS SUBMITTED"
echo "=================================="
echo ""
echo "Job Summary:"
echo "  $JOB1 - no_nucleation plots"
echo "  $JOB2 - gabs overlay"
echo "  $JOB3 - dd overlay (001-020)"
echo "  $JOB4 - dd overlay (021-040)"
echo "  $JOB5 - dd overlay (041-060)"
echo ""