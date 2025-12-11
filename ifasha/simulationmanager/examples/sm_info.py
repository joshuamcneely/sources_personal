# Ada HPC Configuration
# University of York - Ada Supercomputer

# Base paths for uguca_project on Ada
BASE_PROJECT = '/gpfs01/home/pmyjm22/uguca_project'
BASE_SOURCE = '/gpfs01/home/pmyjm22/uguca_project/source'

# Simulation database and build directories
source_dir = f'{BASE_PROJECT}/results'
wdir = f'{BASE_SOURCE}/uguca/build/simulations'
builds_dir = wdir

# Output and processing directories
output_dir = f'{BASE_PROJECT}/results'
temp_dir = f'{BASE_PROJECT}/results'
raw_output_dir = f'{BASE_PROJECT}/results'
postprocess_dir = f'{BASE_PROJECT}/results/processed_data'
fig_dir = f'{BASE_PROJECT}/results/figures'

# Input data directories
input_data_dir = f'{BASE_PROJECT}/results'
restart_dir = f'{BASE_PROJECT}/results'
mesh_dir = f'{BASE_PROJECT}/results'

# Backwards compatibility aliases
base_data_dir = source_dir

"""
Legacy machine configurations (archived):

ubwonko (ga288@ubwonko):
  source_dir = '/home/ga288/sources/database_simulations/'
  wdir = '/home/ga288/builds/simulation-mixed-mode/'
  postprocess_dir = '/home/ga288/postprocess/data/'
  temp_dir = '/home/ga288/raw-output/'
"""
