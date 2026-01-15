#!/usr/bin/env python
"""
Global Simulation Matcher
------------------------------------------------
Compares EVERY Experimental CSV in a folder against EVERY Simulation NPZ in a folder.
Calculates RMSE and ranks the Top 25 matches for each Event.

Usage:
    python compare_all_experiments.py [exp_folder] [sim_folder]
"""

import os
import glob
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg') # Force non-interactive backend for HPC
import matplotlib.pyplot as plt

# Import the plotting function from plot_relative_slip_save.py
try:
    from plot_relative_slip_save import plot_slip_data
except ImportError:
    sys.exit("Error: Could not import 'plot_slip_data' from 'plot_relative_slip_save.py'. Make sure both scripts are in the same directory.")

# --- CONFIGURATION ---
TOP_N = 25  # Number of top matches to display/save per experiment
# ---------------------

def parse_sensor_labels(columns):
    """Parses sensor columns (e.g., 'S1_0.05m') to get positions in meters."""
    positions = []
    col_mapping = {} 
    for i, col in enumerate(columns):
        try:
            parts = col.split('_')
            pos_str = next(p for p in parts if 'm' in p and 'micro' not in p)
            # Handle cases like "0.05m" vs "S1_0.05"
            val = float(pos_str.replace('m', ''))
            positions.append(val)
            col_mapping[val] = i
        except (StopIteration, ValueError):
            pass
    return positions, col_mapping

def estimate_onset(t, U, threshold_ratio=0.10):
    """Estimate onset time based on Peak Slip Rate (Max Gradient)."""
    avg_slip = np.mean(np.abs(U), axis=0)
    try:
        grad = np.gradient(avg_slip)
        idx = np.argmax(grad)
        return t[idx]
    except Exception:
        peak = np.max(avg_slip)
        if peak == 0: return t[0]
        idx = np.argmax(avg_slip > (peak * threshold_ratio))
        return t[idx]

def load_simulation(sim_path):
    try:
        pkg = np.load(sim_path)
        t = pkg['time']
        data = pkg['data'] * 1e6 # Convert Meters -> Microns
        locs = pkg['locations']
        
        # Zero relative slip
        data = data - data[:, 0][:, None]
        
        return t, data, locs
    except Exception:
        return None, None, None

def calculate_match_score(exp_t, exp_data, exp_locs, exp_col_map, sim_t, sim_data, sim_locs):
    # Align via Onset
    t_exp_onset = estimate_onset(exp_t, exp_data)
    t_sim_onset = estimate_onset(sim_t, sim_data)
    shift = t_exp_onset - t_sim_onset
    
    sim_t_shifted = sim_t + shift
    
    all_sq_errors = []
    
    # Compare overlapping sensors (~2cm tolerance)
    for s_idx, s_loc in enumerate(sim_locs):
        dists = [abs(s_loc - el) for el in exp_locs]
        min_dist = min(dists)
        
        if min_dist > 0.02: 
            continue
        
        closest_exp_idx = dists.index(min_dist)
        actual_exp_col_idx = exp_col_map[exp_locs[closest_exp_idx]]
        
        trace_exp = exp_data[actual_exp_col_idx, :]
        trace_sim = sim_data[s_idx, :]
        
        # Interpolate Sim to Exp time
        trace_sim_interp = np.interp(exp_t, sim_t_shifted, trace_sim)
        
        sq_err = (trace_exp - trace_sim_interp)**2
        all_sq_errors.extend(sq_err)
        
    if not all_sq_errors:
        return None  # No matching sensors

    rmse = np.s

def plot_three_way_comparison(exp_path, sim1_data_tuple, sim1_name, sim2_data_tuple, sim2_name, output_dir):
    """
    Plots Experiment vs Original Sim vs Data Driven Sim
    """
    t_sim1, d_sim1, l_sim1 = sim1_data_tuple
    t_sim2, d_sim2, l_sim2 = sim2_data_tuple
    
    # Load Exp
    try:
        df = pd.read_csv(exp_path)
        t_exp = df.iloc[:, 0].values
        exp_sensor_cols = df.columns[1:]
        exp_data = df[exp_sensor_cols].values.T
        exp_data = exp_data - exp_data[:, 0][:, None]
        exp_locs, exp_col_map = parse_sensor_labels(exp_sensor_cols)
    except Exception as e:
        print(f"Error loading exp for plot: {e}")
        return

    # Align Sim1
    t_exp_onset = estimate_onset(t_exp, exp_data)
    if t_sim1 is not None:
        t_sim1_onset = estimate_onset(t_sim1, d_sim1)
        shift1 = t_exp_onset - t_sim1_onset
        t_sim1_shifted = t_sim1 + shift1
    
    # Align Sim2
    if t_sim2 is not None:
        t_sim2_onset = estimate_onset(t_sim2, d_sim2)
        shift2 = t_exp_onset - t_sim2_onset
        t_sim2_shifted = t_sim2 + shift2

    # Plot
    fig, axes = plt.subplots(len(exp_locs), 1, figsize=(10, 3*len(exp_locs)), sharex=True)
    if len(exp_locs) == 1: axes = [axes] [optional: data_driven_sim_folder]")
        sys.exit(1)

    exp_folder = sys.argv[1]
    sim_folder = sys.argv[2]
    dd_sim_folder = sys.argv[3] if len(sys.argv) > 3 else None
    
    # 1. Find Data Files
    exp_files = sorted(glob.glob(os.path.join(exp_folder, "*_displacement.csv")))
    sim_files = sorted(glob.glob(os.path.join(sim_folder, "*.npz")))
    dd_sim_files = sorted(glob.glob(os.path.join(dd_sim_folder, "*.npz"))) if dd_sim_folder else []
    
    if not exp_files:
        print(f"No CSV files found in {exp_folder}")
        sys.exit(1)
    if not sim_files:
        print(f"No NPZ files found in {sim_folder}")
        sys.exit(1)
    if dd_sim_folder and not dd_sim_files:
        print(f"Warning: No NPZ files found in data driven folder {dd_sim_folder}")

    print(f"Found {len(exp_files)} Experiments, {len(sim_files)} Orig sims, {len(dd_sim_files)} Data Driven sims.")
    
    # 2. Pre-load ALL Simulations into Memory (Performance Optimization)
    def cache_sims(files):
        cache = {}
        for f in files:
            sim_name = os.path.basename(f).replace("sim_comparison_", "").replace(".npz", "")
            if 'gabs_fine' in sim_name: continue
            
            t, data, locs = load_simulation(f)
            if t is not None:
                cache[sim_name] = (t, data, locs)
        return cache

    print("Pre-loading original simulation data...")
    sim_cache = cache_sims(sim_files)
    
    dd_sim_cache = {}
    if dd_sim_files:
        print("Pre-loading data driven simulation data...")
        dd_sim_cache = cache_sims(dd_sim_files)
    
    print(f"Loaded {len(sim_cache)} orig sims and {len(dd_sim_cache)} data driven sims.")
    print("Beginning pairwise comparison...")
    print("="*60)

    # Create output directories
    os.makedirs("rankings", exist_ok=True)
    os.makedirs("data_plots", exist_ok=True)
    
    # Open summary file
    with open("rankings/all_top5_matches.txt", "w") as summary_file:
        summary_file.write("Top Matches for Each Experiment\n")
        summary_file.write("="*70 + "\n\n")
        
        # 3. Main Loop
        for exp_path in exp_files:
            exp_name = os.path.basename(exp_path).replace("_displacement.csv", "")
            print(f"\nProcessing: {exp_name}")
            
            # Rank Original Sims
            ranking_orig = process_single_experiment(exp_path, sim_cache)
            # Rank Data Driven Sims
            ranking_dd = process_single_experiment(exp_path, dd_sim_cache)
            
            # --- Reporting ---
            summary_file.write(f"{exp_name}\n")
            summary_file.write("-" * 70 + "\n")
            
            best_orig_name = "None"
            best_orig_data = (None, None, None)
            
            if ranking_orig:
                best_orig_name = ranking_orig[0]['sim']
                best_orig_data = sim_cache[best_orig_name]
                print(f"  Best Orig: {best_orig_name} (RMSE: {ranking_orig[0]['rmse']:.4f})")
                summary_file.write(f"  Best Orig: {best_orig_name} ({ranking_orig[0]['rmse']:.4f})\n")
            
            best_dd_name = "None"
            best_dd_data = (None, None, None)
            
            if ranking_dd:
                best_dd_name = ranking_dd[0]['sim']
                best_dd_data = dd_sim_cache[best_dd_name]
                print(f"  Best Data: {best_dd_name} (RMSE: {ranking_dd[0]['rmse']:.4f})")
                summary_file.write(f"  Best Data: {best_dd_name} ({ranking_dd[0]['rmse']:.4f})\n")
                
            summary_file.write("\n")
            
            # --- Plotting ---
            # If we have at least one match type, plot
            if ranking_orig or ranking_dd:
                plot_three_way_comparison(
                    exp_path, 
                    best_orig_data, best_orig_name,
                    best_dd_data, best_dd_name,
                    "data_plots"
                )

    print("\n" + "="*60)
    print("Done.

    print(f"Found {len(exp_files)} Experiments and {len(sim_files)} Simulations.")
    
    # 2. Pre-load ALL Simulations into Memory (Performance Optimization)
    print("Pre-loading simulation data...")
    sim_cache = {}
    for f in sim_files:
        sim_name = os.path.basename(f).replace("sim_comparison_", "").replace(".npz", "")
        # Filter logic here if strictly needed:
        if 'gabs_fine' in sim_name: continue
        
        t, data, locs = load_simulation(f)
        if t is not None:
            sim_cache[sim_name] = (t, data, locs)
    
    print(f"Loaded {len(sim_cache)} unique simulations.")
    print("Beginning pairwise comparison...")
    print("="*60)

    # Create output directories
    os.makedirs("rankings", exist_ok=True)
    os.makedirs("data_plots", exist_ok=True)
    
    # Open summary file
    with open("rankings/all_top5_matches.txt", "w") as summary_file:
        summary_file.write("Top 5 Matches for Each Experiment\n")
        summary_file.write("="*70 + "\n\n")
        
        # 3. Main Loop
        for exp_path in exp_files:
            exp_name = os.path.basename(exp_path).replace("_displacement.csv", "")
            print(f"\nProcessing: {exp_name}")
            
            ranking = process_single_experiment(exp_path, sim_cache)
            
            # Display Top N
            print(f"{'Rank':<5} | {'Simulation Name':<40} | {'RMSE':<10}")
            print("-" * 65)
            for i, match in enumerate(ranking[:TOP_N]):
                print(f"{i+1:<5} | {match['sim']:<40} | {match['rmse']:.4f}")
            
            # Write top 5 to summary file
            summary_file.write(f"{exp_name}\n")
            summary_file.write("-" * 70 + "\n")
            for i, match in enumerate(ranking[:5]):
                summary_file.write(f"{i+1:2d}. {match['sim']:<40} RMSE: {match['rmse']:8.4f}\n")
            summary_file.write("\n")
            
            # Plot top match using the same plotting function as find_best_match.py
            if ranking:
                top_sim = ranking[0]['sim']
                # Find the sim file path
                sim_file_name = f"sim_comparison_{top_sim}.npz"
                sim_file_path = os.path.join(sim_folder, sim_file_name)
                
                if os.path.exists(sim_file_path):
                    try:
                        plot_slip_data(exp_path, 
                                      sim_path=sim_file_path, 
                                      auto_align=True, 
                                      save_suffix=f"_{top_sim}",
                                      show_plot=False)
                    except Exception as e:
                        print(f"  Warning: Could not plot {exp_name}: {e}")

    print("\n" + "="*60)
    print("Done.")
    print(f"Rankings saved to: rankings/all_top5_matches.txt")
    print(f"Plots saved to: data_plots/")

if __name__ == "__main__":
    main()
