#!/bin/bash
# Submit w_factor comparison job to HPC

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR" || exit 1

echo "[INFO] Submitting w_factor comparison job..."
echo ""

# Submit job
sbatch submit_w_factor_comparison.slurm

echo ""
echo "[INFO] Job submitted. Monitor with: squeue -u \$USER"
echo "[INFO] View log with: tail -f logs/w_factor_comparison_<jobid>.log"
