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
echo "Submitting error jobs..."
echo ""

bash submit_error_jobs.sh

echo ""
echo "Submitting waterfall/overlay jobs..."
echo ""

bash submit_waterfall_jobs.sh

echo ""
echo "=================================="
echo "ALL JOBS SUBMITTED"
echo "=================================="
echo ""
echo "Monitor with: squeue -u $USER"
echo "Check logs in: logs/"
