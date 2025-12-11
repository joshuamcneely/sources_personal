#!/bin/bash
#SBATCH --job-name=postprocess_batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16GB
#SBATCH --time=04:00:00
#SBATCH --partition=compute
#SBATCH --output=postprocess_batch_%j.log
#SBATCH --error=postprocess_batch_%j.err

# Postprocessing batch script for all simulations
# This script processes all simulation results in parallel

set -e

# Setup
PROJECT_ROOT="/gpfs01/home/pmyjm22/uguca_project"
RESULTS_DIR="$PROJECT_ROOT/results"
PPSCRIPTS_DIR="$PROJECT_ROOT/source/postprocessing/ppscripts"
PYTHON_ENV="$PROJECT_ROOT/python_env"

# Activate python environment
source "$PYTHON_ENV/bin/activate"

# Change to ppscripts directory
cd "$PPSCRIPTS_DIR"

# Array of basenames and their sim IDs
declare -A SIMULATIONS=(
  ["rs_slip_asp_XL_v1_0"]="480,485,490"
  ["rs_slip_asp_XLbox_v1_0"]="450,475,500"
  ["rs_slip_asp_v1_0"]="450,475,500"
  ["rs_slip_free_n0_XL_v1_0"]="450,475,500,525"
  ["rs_slip_free_n0_v1_0"]="450,475,500,525"
  ["rs_slip_free_n2_v1_0"]="450,475,500,525"
  ["rs_slip_ref_0"]="450,475,500"
  ["rs_slip_ref_XL_v1_0"]="450,475,500"
  ["rs_slip_ref_v1"]=""
)

# Function to process a single simulation
process_simulation() {
  local basename="$1"
  local sim_id="$2"
  
  echo "Processing $basename with sim_id=$sim_id..."
  
  # Create basename.txt
  echo "$basename" > basename.txt
  
  # Run postprocess in forced mode (non-interactive)
  python3 postprocess.py "$sim_id" forced 2>&1 | tee "postprocess_${basename}_${sim_id}.log"
  
  if [ $? -eq 0 ]; then
    echo "✓ Successfully processed $basename $sim_id"
  else
    echo "✗ Failed to process $basename $sim_id"
    return 1
  fi
}

export -f process_simulation
export PYTHON_ENV RESULTS_DIR PPSCRIPTS_DIR

# Track job count for parallel processing
MAX_PARALLEL=4
JOB_COUNT=0

echo "Starting batch postprocessing..."
echo "========================================"

# Process each basename and its sim IDs
for basename in "${!SIMULATIONS[@]}"; do
  sim_ids="${SIMULATIONS[$basename]}"
  
  # Skip if no sim IDs
  if [ -z "$sim_ids" ]; then
    continue
  fi
  
  # Convert comma-separated to array
  IFS=',' read -ra sim_array <<< "$sim_ids"
  
  for sim_id in "${sim_array[@]}"; do
    # Run in background and manage parallel jobs
    process_simulation "$basename" "$sim_id" &
    
    JOB_COUNT=$((JOB_COUNT + 1))
    
    # Wait if we've reached max parallel jobs
    if [ $JOB_COUNT -ge $MAX_PARALLEL ]; then
      wait -n  # Wait for any job to complete
      JOB_COUNT=$((JOB_COUNT - 1))
    fi
  done
done

# Wait for all remaining background jobs
wait

echo "========================================"
echo "Batch postprocessing complete!"
echo ""
echo "Check logs for details:"
echo "  postprocess_batch_*.log"
echo "  postprocess_*_*.log"
