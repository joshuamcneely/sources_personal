#!/bin/bash
# Submit postprocessing jobs, one per basename
# Each job sets its own basename.txt and processes all sim IDs for that basename

PROJECT_ROOT="/gpfs01/home/pmyjm22/uguca_project"
PPSCRIPTS_DIR="$PROJECT_ROOT/source/postprocessing/ppscripts"

cd "$PPSCRIPTS_DIR"

# Define all basenames and their sim IDs
declare -A SIMULATIONS=(
  ["rs_slip_asp_XL_v1_0"]="480,485,490"
  ["rs_slip_asp_XLbox_v1_0"]="450,475,500"
  ["rs_slip_asp_v1_0"]="450,475,500"
  ["rs_slip_free_n0_XL_v1_0"]="450,475,500,525"
  ["rs_slip_free_n0_v1_0"]="450,475,500,525"
  ["rs_slip_free_n2_v1_0"]="450,475,500,525"
  ["rs_slip_ref_0"]="450,475,500"
  ["rs_slip_ref_XL_v1_0"]="450,475,500"
)

echo "Submitting postprocessing jobs..."
echo "========================================"

for basename in "${!SIMULATIONS[@]}"; do
  sim_ids="${SIMULATIONS[$basename]}"
  
  # Skip if no sim IDs
  if [ -z "$sim_ids" ]; then
    continue
  fi
  
  echo "Submitting job for $basename (sim IDs: $sim_ids)"
  sbatch --export=BASENAME="$basename",SIMIDS="$sim_ids" postprocess_single_basename.sh
done

echo "========================================"
echo "All jobs submitted!"
echo ""
echo "Check status with: squeue -u pmyjm22"
echo "View logs: postprocess_*_*.log"
