#!/bin/bash
#SBATCH --job-name=extract_gabs_array
#SBATCH --output=logs/extract_%A_%a.log
#SBATCH --error=logs/extract_%A_%a.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:30:00
#SBATCH --mem=4G
#SBATCH --array=1-162%20   # Process all 162 simulations

# Create logs directory if it doesn't exist
mkdir -p logs

# Get the directory listing into an array
SIM_DIRS=(data/gabs_fine_*.datamanager.info)
NUM_SIMS=${#SIM_DIRS[@]}

# SLURM_ARRAY_TASK_ID is 1-based, array is 0-based
INDEX=$(($SLURM_ARRAY_TASK_ID - 1))

if [ $INDEX -ge $NUM_SIMS ]; then
    echo "Index $INDEX out of bounds (max $NUM_SIMS)"
    exit 0
fi

DIR_PATH="${SIM_DIRS[$INDEX]}"
FILENAME=$(basename "$DIR_PATH")
SIM_NAME="${FILENAME%.datamanager.info}"

echo "Task ID: $SLURM_ARRAY_TASK_ID processing $SIM_NAME"

# Run the python extraction script
python extract_sim_for_overlay.py "$SIM_NAME"
