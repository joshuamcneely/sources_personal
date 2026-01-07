#!/bin/bash
#SBATCH --job-name=extract_gabs
#SBATCH --output=extract_gabs_%j.log
#SBATCH --error=extract_gabs_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=04:00:00
#SBATCH --mem=8G

# Activate your python environment (adjust if necessary)
# source activate python_env 
# OR just rely on the environment being active when submitted effectively via the shell

echo "Starting extraction job..."
cd $SLURM_SUBMIT_DIR

# Ensure the extraction script is executable
chmod +x extract_all_gabs.sh

# Run the extraction script
./extract_all_gabs.sh

echo "Job finished."
