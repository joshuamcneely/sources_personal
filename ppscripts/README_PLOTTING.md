# Batch Plotting Workflow

Complete workflow for generating all comparison plots between simulations and experiments.

## Directory Structure
```
ppscripts/
├── sanity_check_plots.sh          # Test script - run first!
├── submit_all_plots.sh             # Master submission script
├── submit_error_gabs.slurm         # Error: gabs_fine (10 sims)
├── submit_error_dd_01-20.slurm     # Error: dd_study 001-020
├── submit_error_dd_21-40.slurm     # Error: dd_study 021-040
├── submit_error_dd_41-60.slurm     # Error: dd_study 041-060
├── submit_plot_no_nucleation.slurm # Plots: dd_no_nucleation_data_only
├── submit_overlay_gabs.slurm       # Overlay: gabs_fine
├── submit_overlay_dd_01-20.slurm   # Overlay: dd_study 001-020
├── submit_overlay_dd_21-40.slurm   # Overlay: dd_study 021-040
├── submit_overlay_dd_41-60.slurm   # Overlay: dd_study 041-060
└── merge_error_results.py          # Post-processing: merge CSV results
```

## Quick Start

### Step 1: Sanity Check (ALWAYS RUN FIRST)
```bash
cd /gpfs01/home/pmyjm22/uguca_project/source/postprocessing/ppscripts
bash sanity_check_plots.sh
```

If sanity check passes, proceed to Step 2. If it fails, debug before submitting jobs.

### Step 2: Submit All Jobs
```bash
bash submit_all_plots.sh
```

### Step 3: Monitor Progress
```bash
# Check job status
squeue -u $USER

# Watch logs in real-time (example)
tail -f logs/error_gabs_*.log

# Check for errors
grep -i error logs/*.err
```

### Step 4: Merge Results (After Jobs Complete)
```bash
python merge_error_results.py
```

## Output Locations

### Error Analysis Plots
- `plots/error_plots/gabs_fine/` - Error plots for gabs_fine simulations
- `plots/error_plots/dd_study_01-20/` - Error plots for dd 001-020
- `plots/error_plots/dd_study_21-40/` - Error plots for dd 021-040
- `plots/error_plots/dd_study_41-60/` - Error plots for dd 041-060
- `plots/error_plots/dd_study_ALL_error_results.csv` - Merged results (001-060)

### Overlay Waterfalls
- `plots/overlay/gabs_fine_multi_overlay.png` - All gabs_fine + experiment
- `plots/overlay/dd_study_01-20_multi_overlay.png` - dd 001-020 + experiment
- `plots/overlay/dd_study_21-40_multi_overlay.png` - dd 021-040 + experiment
- `plots/overlay/dd_study_41-60_multi_overlay.png` - dd 041-060 + experiment

### Special Plots
- `plots/no_nucleation/` - All plots for dd_no_nucleation_data_only

## Individual Job Submission

If you need to run jobs individually:

```bash
# Error analysis
sbatch submit_error_gabs.slurm
sbatch submit_error_dd_01-20.slurm

# Overlays
sbatch submit_overlay_gabs.slurm

# Special cases
sbatch submit_plot_no_nucleation.slurm
```

## Debugging

### Check if simulations are processed
```bash
ls -la data/ | grep gabs_fine
ls -la data/ | grep dd_exp_w_factor_study
```

### Test one simulation manually
```bash
# Error calculation
python calculate_slip_error_norms.py gabs_fine_124 data_experiments/FS01-043-4MPa-RP-1_Event_1_displacement.csv

# Waterfall plot
python waterfall_and_space_time_plot.py gabs_fine_124 interface auto
```

### Common Issues

**Job fails immediately:**
- Check that data directory exists and is readable
- Verify experimental CSV path is correct
- Run sanity check script first

**Job runs but produces no output:**
- Check logs in `logs/` directory
- Verify simulation names match processed data
- Check memory limits (increase if needed)

**Plots missing:**
- Ensure `plots/` directory permissions are correct
- Check matplotlib backend is set correctly (Agg)
- Review job output logs

## Job Resource Summary

| Job | Time | Memory | CPUs | Note |
|-----|------|--------|------|------|
| error_gabs | 2h | 8G | 2 | 10 simulations |
| error_dd_01-20 | 4h | 16G | 4 | 20 simulations |
| error_dd_21-40 | 4h | 16G | 4 | 20 simulations |
| error_dd_41-60 | 4h | 16G | 4 | 20 simulations |
| plot_no_nuc | 2h | 8G | 2 | Single sim + error |
| overlay_gabs | 3h | 12G | 4 | 10 sims overlay |
| overlay_dd_01-20 | 4h | 16G | 4 | 20 sims overlay |
| overlay_dd_21-40 | 4h | 16G | 4 | 20 sims overlay |
| overlay_dd_41-60 | 4h | 16G | 4 | 20 sims overlay |

Total: ~9 jobs, max 4 hours each
