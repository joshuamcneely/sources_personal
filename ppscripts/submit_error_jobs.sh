#!/bin/bash
# Submit only error analysis jobs

echo "=================================="
echo "SUBMITTING ERROR JOBS"
echo "=================================="

# Create directories
mkdir -p logs
mkdir -p plots/error_plots

echo ""
echo "Submitting error jobs in sequence..."
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

echo ""
echo "=================================="
echo "ERROR JOBS SUBMITTED"
echo "=================================="
echo ""
echo "Job Summary:"
echo "  $JOB1 - gabs error"
echo "  $JOB2 - dd error (001-020)"
echo "  $JOB3 - dd error (021-040)"
echo "  $JOB4 - dd error (041-060)"
echo ""