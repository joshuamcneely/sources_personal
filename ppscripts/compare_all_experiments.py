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
    if len(exp_locs) == 1: axes = [axes]
    
    for i, ax in enumerate(axes):
        loc = exp_locs[i]
        # Exp Trace
        col_idx = exp_col_map[loc]
        ax.plot(t_exp, exp_data[col_idx], 'k-', label='Experiment', linewidth=2, alpha=0.6)
        
        # Sim1 Trace
        if t_sim1 is not None:
            dists1 = [abs(loc - sl) for sl in l_sim1]
            if min(dists1) < 0.02:
                idx1 = dists1.index(min(dists1))
                ax.plot(t_sim1_shifted, d_sim1[idx1], 'b--', label=f'Orig: {sim1_name}', linewidth=1.5)

        # Sim2 Trace
        if t_sim2 is not None:
            dists2 = [abs(loc - sl) for sl in l_sim2]
            if min(dists2) < 0.02:
                idx2 = dists2.index(min(dists2))
                ax.plot(t_sim2_shifted, d_sim2[idx2], 'r-.', label=f'Data: {sim2_name}', linewidth=1.5)
            
        ax.set_ylabel(f"Slip ($\mu m$) @ {loc}m")
        ax.grid(True, linestyle=':', alpha=0.6)
        if i == 0:
            ax.legend()
            ax.set_title(f"{os.path.basename(exp_path)}")
            
    axes[-1].set_xlabel("Time (s)")
    plt.tight_layout()
    plot_name = os.path.basename(exp_path).replace('.csv', '') + '_comparison.png'
    plt.savefig(os.path.join(output_dir, plot_name), dpi=150)
    plt.close()

def plot_family_comparison(exp_path, exp_name_core, sim_cache, dd_sim_cache, output_dir):
    """
    Plots:
    1. Experiment (Black)
    2. 'gabs_fine' Original Sims that this experiment's DD runs are based on (Blue dashed)
    3. ALL Data Driven runs associated with this experiment (Red/Orange lines)
    """
    # 1. Start Figure
    try:
        df = pd.read_csv(exp_path)
        t_exp = df.iloc[:, 0].values
        exp_sensor_cols = df.columns[1:]
        exp_data = df[exp_sensor_cols].values.T
        exp_data = exp_data - exp_data[:, 0][:, None]
        exp_locs, exp_col_map = parse_sensor_labels(exp_sensor_cols)
    except Exception as e:
        print(f"Error loading exp for family plot: {e}")
        return

    # Align Exp Onset
    t_exp_onset = estimate_onset(t_exp, exp_data)
    
    # 2. Identify Relevant DD Runs
    # Looking for keys in dd_sim_cache that start with "dd_exp_{exp_name_core}"
    relevant_dd_keys = [k for k in dd_sim_cache.keys() if f"dd_exp_{exp_name_core}" in k]
    
    if not relevant_dd_keys:
        return # No family to plot
        
    # 3. Identify Relevant Original Sims base on DD names
    # Key format: dd_exp_EXPNAME_vs_ORIGSIMNAME
    relevant_orig_keys = set()
    for k in relevant_dd_keys:
        # A bit of a hacky parse. Let's assume structure is consistently _vs_
        # Example: dd_exp_FS01-...-Event3_vs_gabs_rep_2
        parts = k.split("_vs_")
        if len(parts) > 1:
            # We need to clean sim name. Usually just the part after vs
            # But the key in sim_cache was cleaned during loading.
            # Usually: gabs_rep_2
            candidate = parts[1] 
            # It might have suffixes from the original filename cleaning?
            
            # Let's search for partial match in sim_cache keys
            for sim_key in sim_cache.keys():
                if candidate in sim_key or sim_key in candidate:
                     relevant_orig_keys.add(sim_key)

    # 4. Plot Setup
    fig, axes = plt.subplots(len(exp_locs), 1, figsize=(10, 3*len(exp_locs)), sharex=True)
    if len(exp_locs) == 1: axes = [axes]

    for i, ax in enumerate(axes):
        loc = exp_locs[i]
        col_idx = exp_col_map[loc]
        
        # Plot Experiment
        ax.plot(t_exp, exp_data[col_idx], 'k-', label='Experiment' if i==0 else "", linewidth=2.5, zorder=10)
        
        # Plot Original Sims (Blue)
        for orig_name in relevant_orig_keys:
            if orig_name not in sim_cache: continue
            t, d, l = sim_cache[orig_name]
            
            dists = [abs(loc - sl) for sl in l]
            min_dist = min(dists)
            if min_dist < 0.02:
                idx = dists.index(min_dist)
                # Align
                t_onset = estimate_onset(t, d)
                t_shift = t + (t_exp_onset - t_onset)
                
                label = f"Orig: {orig_name}" if i==0 else ""
                ax.plot(t_shift, d[idx], 'b--', label=label, linewidth=1.5, alpha=0.7)

        # Plot DD Sims (Red/Orange)
        cmap = plt.get_cmap('autumn')
        for j, dd_name in enumerate(relevant_dd_keys):
            t, d, l = dd_sim_cache[dd_name]
            
            dists = [abs(loc - sl) for sl in l]
            min_dist = min(dists)
            if min_dist < 0.02:
                idx = dists.index(min_dist)
                # Align
                t_onset = estimate_onset(t, d)
                t_shift = t + (t_exp_onset - t_onset)
                
                color = cmap(j / len(relevant_dd_keys))
                label = "Data Driven" if (i==0 and j==0) else ""
                ax.plot(t_shift, d[idx], color=color, linestyle='-.', label=label, linewidth=1.0, alpha=0.8)

        ax.set_ylabel(f"Slip ($\mu m$) @ {loc}m")
        ax.grid(True, linestyle=':', alpha=0.6)
        if i == 0:
            ax.legend()
            ax.set_title(f"Family: {os.path.basename(exp_path)}")

    axes[-1].set_xlabel("Time (s)")
    plt.tight_layout()
    plot_name = os.path.basename(exp_path).replace('.csv', '') + '_family_comparison.png'
    plt.savefig(os.path.join(output_dir, plot_name), dpi=150)
    plt.close()

def main():
    if len(sys.argv) < 3:
        print("Usage: python compare_all_experiments.py [exp_folder] [sim_folder]
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
            # If we have at least one match type, plot Best Match
            if ranking_orig or ranking_dd:
                plot_three_way_comparison(
                    exp_path, 
                    best_orig_data, best_orig_name,
                    best_dd_data, best_dd_name,
                    "data_plots"
                )
            
            # --- Family Plotting ---
            # Plot all DD runs associated with this specific experiment vs the orig sim they used
            if dd_sim_files:
                plot_family_comparison(
                    exp_path,
                    exp_name,
                    sim_cache,
                    dd_sim_cache,
                    "data_plots"
                )

    print("\n" + "="*60)
    print("Done.")

if __name__ == "__main__":
    main()
