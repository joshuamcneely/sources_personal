# Ada HPC Configuration - COMPLETE

All hardcoded paths have been updated for the Ada supercomputer at University of York.

## Configuration Summary

**Base paths on Ada:**
- Project root: `/gpfs01/home/pmyjm22/uguca_project`
- Source code: `/gpfs01/home/pmyjm22/uguca_project/source`
- ppscripts location: `/gpfs01/home/pmyjm22/uguca_project/source/sources`

## Updated Files

### 1. `ifasha/simulationmanager/examples/sm_info.py`
**What changed:** Replaced environment variable-based configuration with Ada-specific paths.

**Active paths set:**
- `source_dir` → `/gpfs01/home/pmyjm22/uguca_project/source/uguca/simulations`
- `wdir` (working directory) → `/gpfs01/home/pmyjm22/uguca_project/source/uguca/build/simulations`
- `builds_dir` → `/gpfs01/home/pmyjm22/uguca_project/source/uguca/build/simulations`
- `output_dir` → `/gpfs01/home/pmyjm22/uguca_project/results`
- `temp_dir` → `/gpfs01/home/pmyjm22/uguca_project/results`
- `raw_output_dir` → `/gpfs01/home/pmyjm22/uguca_project/results`
- `postprocess_dir` → `/gpfs01/home/pmyjm22/uguca_project/results/processed_data`
- `fig_dir` → `/gpfs01/home/pmyjm22/uguca_project/results/figures`
- `input_data_dir` → `/gpfs01/home/pmyjm22/uguca_project/source/uguca/simulations/input_files`
- `restart_dir` → `/gpfs01/home/pmyjm22/uguca_project/source/uguca/simulations/input_files/restart`
- `mesh_dir` → `/gpfs01/home/pmyjm22/uguca_project/source/uguca/simulations/input_files/meshes`

### 2. `ifasha/simulationmanager/examples/sm_generatestatic.py`
**What changed:** Updated PBS job submission script template.
- Changed email placeholder from `ga288@cornell.edu` to `{email}` (to be filled in at runtime)
- Changed module loading from `/home/ga288/module-list-ubwonko.sh` to `/gpfs01/home/pmyjm22/uguca_project/source/uguca/module-list.sh`

### 3. `ifasha/simulationmanager/examples/sm_generatedynamic.py`
**What changed:** Updated PBS job submission script template.
- Changed email placeholder from `ga288@cornell.edu` to `{email}` (to be filled in at runtime)
- Changed module loading from `/home/ga288/module-list-ubwonko.sh` to `/gpfs01/home/pmyjm22/uguca_project/source/uguca/module-list.sh`

### 4. `ppscripts/source_directories.txt`
**What changed:** Updated output directory reference.
- Old: `/Users/joshmcneely/introsims/simulation_outputs`
- New: `/gpfs01/home/pmyjm22/uguca_project/results`

## Next Steps

1. **Verify directories exist on Ada:**
   ```bash
   ls -la /gpfs01/home/pmyjm22/uguca_project/source/uguca/simulations/input_files/
   ls -la /gpfs01/home/pmyjm22/uguca_project/results/
   ```

2. **Create output subdirectories if needed:**
   ```bash
   mkdir -p /gpfs01/home/pmyjm22/uguca_project/results/processed_data
   mkdir -p /gpfs01/home/pmyjm22/uguca_project/results/figures
   ```

3. **Verify module script exists:**
   ```bash
   ls -la /gpfs01/home/pmyjm22/uguca_project/source/uguca/module-list.sh
   ```

4. **Test imports in Python:**
   ```python
   import sys
   sys.path.insert(0, '/gpfs01/home/pmyjm22/uguca_project/source/sources/ifasha/simulationmanager/examples')
   import sm_info
   print(sm_info.source_dir)  # Should print Ada paths
   ```

## Status: ✅ Ready for Ada HPC

All hardcoded machine-specific paths have been replaced with Ada HPC paths. The codebase is now configured for your Ada account.
