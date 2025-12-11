#!/bin/bash
#SBATCH --job-name=postprocess_%x
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16GB
#SBATCH --time=02:00:00
#SBATCH --partition=defq
#SBATCH --output=postprocess_%x_%j.log
#SBATCH --error=postprocess_%x_%j.err

# Single-basename postprocessing job
# Submitted by postprocess_submit_all.sh
# Usage: sbatch --export=BASENAME=rs_slip_asp_XL_v1_0,SIMIDS="480,485,490" postprocess_single_basename.sh

set -e

# Setup
PROJECT_ROOT="/gpfs01/home/pmyjm22/uguca_project"
PPSCRIPTS_DIR="$PROJECT_ROOT/source/postprocessing/ppscripts"
PYTHON_ENV="$PROJECT_ROOT/python_env"

# Activate python environment
source "$PYTHON_ENV/bin/activate"

# Change to ppscripts directory
cd "$PPSCRIPTS_DIR"

# Sanity checks
if [ -z "$BASENAME" ] || [ -z "$SIMIDS" ]; then
  echo "Error: BASENAME and SIMIDS must be set"
  exit 1
fi

echo "Processing basename: $BASENAME"
echo "Sim IDs: $SIMIDS"
echo "========================================"

# Set basename.txt once for all sims in this job
echo "$BASENAME" > basename.txt

# Convert comma-separated sim IDs to array
IFS=',' read -ra sim_array <<< "$SIMIDS"

# Process each sim ID
for sim_id in "${sim_array[@]}"; do
  echo ""
  echo "Processing sim_id=$sim_id..."
  
  # Run postprocess in forced mode (non-interactive)
  if python3 postprocess.py "$sim_id" forced 2>&1 | tee "postprocess_${BASENAME}_${sim_id}.log"; then
    echo "✓ Successfully processed $BASENAME $sim_id"
  else
    echo "✗ Failed to process $BASENAME $sim_id"
  fi
done

echo ""
echo "========================================"
echo "Batch complete for $BASENAME"
