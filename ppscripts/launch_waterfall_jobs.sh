#!/bin/bash

# 1. Create logs directory (Slurm requires this to exist before starting)
mkdir -p logs

# 2. Count files to determine array size
NUM_FILES=$(ls data/gabs_fine_*.datamanager.info | wc -l)

# Arrays are 0-indexed, so we need 0 to (N-1)
MAX_ID=$((NUM_FILES - 1))

echo "Found $NUM_FILES simulations."
echo "Submitting job array for indices [0 - $MAX_ID]..."

# 3. Submit
sbatch --array=0-$MAX_ID submit_waterfall.slurm
