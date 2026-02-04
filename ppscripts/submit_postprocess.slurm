#!/bin/bash
#SBATCH --job-name=postprocess_dd_no_nucleation
#SBATCH --time=04:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --output=postprocess.log
#SBATCH --error=postprocess.err

cd /gpfs01/home/pmyjm22/uguca_project/ppscripts

# Set empty basename for simulations without numeric ID
echo "" > basename.txt

# Run postprocessing
python postprocess.py dd_no_nucleation_data_only
