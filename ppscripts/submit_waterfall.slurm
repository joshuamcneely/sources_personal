#!/bin/bash
#SBATCH --job-name=viz_waterfall
#SBATCH --output=logs/viz_%A_%a.out
#SBATCH --error=logs/viz_%A_%a.err
#SBATCH --time=00:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G

# --- 1. Setup ---
# Ensure we are in the right directory
cd /gpfs01/home/pmyjm22/uguca_project/source/postprocessing/ppscripts

# --- 2. Get File for this Task ID ---
# Create an array of all matching files
FILES=(data/dd_exp_w_factor_study_*.datamanager.info)

# Select the specific file for this sub-job using the Slurm Array ID
CURRENT_FILE="${FILES[$SLURM_ARRAY_TASK_ID]}"

# Extract the simulation name (e.g., "dd_exp_w_factor_study_001")
SNAME=$(basename "$CURRENT_FILE" .datamanager.info)

echo "------------------------------------------------"
echo "Job Array ID: $SLURM_ARRAY_TASK_ID"
echo "Processing:   $SNAME"
echo "------------------------------------------------"

# --- 3. Experimental Data Path ---
EXP_CSV="data_experiments/FS01-043-4MPa-RP-1_Event_1_displacement.csv"

# --- 4. Run Python ---
# Note: This assumes your python environment is active or 'python3' is in path.
# If you need to load modules or activate a venv, add it here:
# source /path/to/my/venv/bin/activate

python3 waterfall_and_space_time_plot.py "$SNAME" interface top_disp "$EXP_CSV"
