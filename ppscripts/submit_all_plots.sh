#!/bin/bash
# Master submission script for all plotting tasks

echo "=================================="
echo "SUBMITTING ALL PLOTTING JOBS"
echo "=================================="

# Create directories
mkdir -p logs
mkdir -p plots/error_plots
mkdir -p plots/overlay
mkdir -p plots/no_nucleation

echo ""
echo "Submitting jobs in sequence..."
echo ""

# Task 1: gabs_fine error analysis
echo "1. Submitting gabs_fine error analysis..."
JOB1=$(sbatch submit_error_gabs.slurm | awk '{print $4}')
echo "   Job ID: $JOB1"

# Task 2: dd_exp_w_factor_study error analysis (3 batches)
echo "2. Submitting dd_study error analysis (001-020)..."
JOB2=$(sbatch submit_error_dd_01-20.slurm | awk '{print $4}')
echo "   Job ID: $JOB2"

echo "3. Submitting dd_study error analysis (021-040)..."
JOB3=$(sbatch submit_error_dd_21-40.slurm | awk '{print $4}')
echo "   Job ID: $JOB3"

echo "4. Submitting dd_study error analysis (041-060)..."
JOB4=$(sbatch submit_error_dd_41-60.slurm | awk '{print $4}')
echo "   Job ID: $JOB4"

# Task 3: dd_no_nucleation_data_only plots
echo "5. Submitting dd_no_nucleation_data_only plots..."
JOB5=$(sbatch submit_plot_no_nucleation.slurm | awk '{print $4}')
echo "   Job ID: $JOB5"

# Task 4: gabs_fine overlay
echo "6. Submitting gabs_fine multi-overlay..."
JOB6=$(sbatch submit_overlay_gabs.slurm | awk '{print $4}')
echo "   Job ID: $JOB6"

# Task 5: dd_exp_w_factor_study overlays (3 batches)
echo "7. Submitting dd_study multi-overlay (001-020)..."
JOB7=$(sbatch submit_overlay_dd_01-20.slurm | awk '{print $4}')
echo "   Job ID: $JOB7"

echo "8. Submitting dd_study multi-overlay (021-040)..."
JOB8=$(sbatch submit_overlay_dd_21-40.slurm | awk '{print $4}')
echo "   Job ID: $JOB8"

echo "9. Submitting dd_study multi-overlay (041-060)..."
JOB9=$(sbatch submit_overlay_dd_41-60.slurm | awk '{print $4}')
echo "   Job ID: $JOB9"

echo ""
echo "=================================="
echo "ALL JOBS SUBMITTED"
echo "=================================="
echo ""
echo "Job Summary:"
echo "  $JOB1 - gabs error"
echo "  $JOB2 - dd error (001-020)"
echo "  $JOB3 - dd error (021-040)"
echo "  $JOB4 - dd error (041-060)"
echo "  $JOB5 - no_nucleation plots"
echo "  $JOB6 - gabs overlay"
echo "  $JOB7 - dd overlay (001-020)"
echo "  $JOB8 - dd overlay (021-040)"
echo "  $JOB9 - dd overlay (041-060)"
echo ""
echo "Monitor with: squeue -u $USER"
echo "Check logs in: logs/"
