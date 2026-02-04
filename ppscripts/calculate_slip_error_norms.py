#!/usr/bin/env python
"""
Calculate L2 and L-infinity norms between simulation and experimental slip data.

This script compares simulation displacement (doubled for top_disp) against experimental
slip measurements at sensor locations.

Usage:
    python calculate_slip_error_norms.py <sim_name> <exp_csv> [--group interface] [--sensor-range 0.0 3.0]
    
    OR for batch comparison:
    python calculate_slip_error_norms.py --batch sim1,sim2,sim3 exp.csv [--output results.csv]
"""
from __future__ import print_function, division, absolute_import

import sys
import os
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for HPC
import matplotlib.pyplot as plt
import ifasha.datamanager as idm
import ppscripts.postprocess as pp

# Sensor positions (McKlaskey layout: 0.05 to 3.05 m, 0.2 m spacing)
DEFAULT_SENSOR_X = np.arange(0.05, 3.05 + 0.01, 0.2)

def estimate_onset(t, U, threshold_ratio=0.1):
    """Estimate onset time based on peak slip rate (max gradient)."""
    avg_slip = np.mean(np.abs(U), axis=0)
    try:
        grad = np.gradient(avg_slip)
        idx = np.argmax(grad)
        return t[idx]
    except Exception:
        peak = np.max(avg_slip)
        if peak == 0:
            return t[0]
        idx = np.argmax(avg_slip > (peak * threshold_ratio))
        return t[idx]


def select_disp_field(data):
    """Choose a displacement field that exists in the dataset."""
    candidates = [
        idm.FieldId('top_disp', 0),
        idm.FieldId('top_disp', 1),
        idm.FieldId('top_disp'),
        idm.FieldId('interface_top_disp', 0),
        idm.FieldId('interface_top_disp', 1),
        idm.FieldId('disp', 0),
        idm.FieldId('disp', 1),
    ]
    for fid in candidates:
        if data.has_field(fid):
            return fid
    raise RuntimeError('No displacement field found in dataset')


def load_experimental_data(csv_path, sensor_x=DEFAULT_SENSOR_X):
    """
    Load experimental CSV and return aligned time and slip data.
    Returns: (time_aligned, slip_microns, sensor_positions)
    """
    if not os.path.exists(csv_path):
        raise RuntimeError("Experimental CSV not found: {}".format(csv_path))
    
    df = pd.read_csv(csv_path)
    time_col = df.columns[0]
    sensor_cols = df.columns[1:]
    
    t_full = df[time_col].values
    U_raw = df[sensor_cols].values.T  # (n_sensors, n_times)
    
    # Relative slip (zero to first sample)
    U_rel = U_raw - U_raw[:, [0]]
    
    # Detect onset
    t_onset = estimate_onset(t_full, U_rel)
    t_aligned = t_full - t_onset
    
    print("  Experimental onset: {:.6f} s".format(t_onset))
    print("  Time range: [{:.6f}, {:.6f}] s".format(t_aligned[0], t_aligned[-1]))
    print("  {} sensors, {} time points".format(U_rel.shape[0], U_rel.shape[1]))
    
    return t_aligned, U_rel, sensor_x[:U_rel.shape[0]]


def load_simulation_data(sname, group, sensor_x=DEFAULT_SENSOR_X, wdir='./data'):
    """
    Load simulation data at sensor positions.
    Returns: (time_aligned, slip_microns, sensor_positions)
    """
    sname = pp.sname_to_sname(sname)
    dma = idm.DataManagerAnalysis(sname, wdir)
    
    try:
        data = dma(group)
    except RuntimeError:
        raise RuntimeError("Group '{}' not found in simulation '{}'".format(group, sname))
    
    # Get positions
    if data.has_field(idm.FieldId('position', 0)):
        pos_fid = idm.FieldId('position', 0)
    elif data.has_field(idm.FieldId('coord', 0)):
        pos_fid = idm.FieldId('coord', 0)
    else:
        raise RuntimeError('No position/coord field found')
    
    positions = np.array(data.get_field_at_t_index(pos_fid, 0)[0])
    if positions.ndim > 1:
        coords_x = positions[:, 0]
    else:
        coords_x = positions
    
    # Map sensors to nearest node indices
    node_indices = []
    actual_positions = []
    for sx in sensor_x:
        idx = int(np.argmin(np.abs(coords_x - sx)))
        node_indices.append(idx)
        actual_positions.append(coords_x[idx])
    
    # Get displacement field
    disp_fid = select_disp_field(data)
    slip_factor = 2.0 if disp_fid.name.startswith('top_disp') else 1.0
    
    print("  Using field: {} (slip_factor={})".format(disp_fid.get_string(), slip_factor))
    
    # Extract time array
    time_fid = idm.FieldId("time")
    last_idx = data.get_t_index("last")
    time_steps = np.arange(last_idx + 1)
    
    times = []
    for t_idx in time_steps:
        t_val_container = data.get_field_at_t_index(time_fid, t_idx)[0]
        t_val = t_val_container[0] if isinstance(t_val_container, (np.ndarray, list)) else t_val_container
        times.append(t_val)
    times = np.array(times)
    
    # Extract displacement at sensor locations
    disp = np.zeros((len(sensor_x), len(times)))
    for j, n_idx in enumerate(node_indices):
        vals = []
        for t_idx in time_steps:
            snap = data.get_field_at_t_index(disp_fid, t_idx)[0]
            vals.append(snap[n_idx])
        series = np.array(vals)
        # Convert to microns and zero to first sample
        series = (series - series[0]) * slip_factor * 1e6
        disp[j, :] = series
    
    # Align onset to t=0
    t_onset = estimate_onset(times, disp)
    times_aligned = times - t_onset
    
    print("  Simulation onset: {:.6f} s".format(t_onset))
    print("  Time range: [{:.6f}, {:.6f}] s".format(times_aligned[0], times_aligned[-1]))
    print("  {} sensors, {} time points".format(disp.shape[0], disp.shape[1]))
    
    return times_aligned, disp, np.array(actual_positions)


def compute_error_norms(sim_t, sim_data, exp_t, exp_data, t_min=None, t_max=None, use_exp_times=True):
    """
    Compute L2 and L-infinity norms between simulation and experiment.
    
    Args:
        sim_t: simulation time array (aligned)
        sim_data: simulation slip (n_sensors, n_times) in microns
        exp_t: experimental time array (aligned)
        exp_data: experimental slip (n_sensors, n_times) in microns
        t_min, t_max: optional time window for comparison
        use_exp_times: if True, only compare at experimental time points (more efficient)
    
    Returns:
        dict with L2 norm, L-infinity norm, RMSE per sensor, max error per sensor
    """
    n_sensors = min(sim_data.shape[0], exp_data.shape[0])
    
    # Determine overlap window
    t_start = max(sim_t[0], exp_t[0])
    t_end = min(sim_t[-1], exp_t[-1])
    
    if t_min is not None:
        t_start = max(t_start, t_min)
    if t_max is not None:
        t_end = min(t_end, t_max)
    
    print("\n  Comparison window: [{:.6f}, {:.6f}] s".format(t_start, t_end))
    
    if use_exp_times:
        # Compare only at experimental time points (exact comparison, no interpolation)
        exp_mask = (exp_t >= t_start) & (exp_t <= t_end)
        common_t = exp_t[exp_mask]
        
        print("  Using experimental time points: {} points".format(len(common_t)))
        
        # Interpolate simulation to experimental times
        sim_interp = np.zeros((n_sensors, len(common_t)))
        exp_interp = np.zeros((n_sensors, len(common_t)))
        
        for i in range(n_sensors):
            sim_interp[i, :] = np.interp(common_t, sim_t, sim_data[i, :])
            exp_interp[i, :] = exp_data[i, exp_mask]
    else:
        # Use dense common grid (original behavior)
        dt = np.median(np.diff(exp_t))  # Use experimental resolution
        common_t = np.arange(t_start, t_end + dt/2, dt)
        
        print("  Common time grid: {} points at dt={:.2e} s".format(len(common_t), dt))
        
        # Interpolate data
        sim_interp = np.zeros((n_sensors, len(common_t)))
        exp_interp = np.zeros((n_sensors, len(common_t)))
        
        for i in range(n_sensors):
            sim_interp[i, :] = np.interp(common_t, sim_t, sim_data[i, :])
            exp_interp[i, :] = np.interp(common_t, exp_t, exp_data[i, :])
    
    # Compute errors
    error = sim_interp - exp_interp
    
    # L2 norm (RMS error across all sensors and time)
    l2_norm = np.sqrt(np.mean(error**2))
    
    # L-infinity norm (max absolute error)
    linf_norm = np.max(np.abs(error))
    
    # Per-sensor metrics
    rmse_per_sensor = np.sqrt(np.mean(error**2, axis=1))
    max_error_per_sensor = np.max(np.abs(error), axis=1)
    
    return {
        'l2_norm': l2_norm,
        'linf_norm': linf_norm,
        'rmse_per_sensor': rmse_per_sensor,
        'max_error_per_sensor': max_error_per_sensor,
        'n_sensors': n_sensors,
        'n_time_points': len(common_t),
        'time_window': (t_start, t_end)
    }


def print_results(sim_name, results):
    """Pretty print error norm results."""
    print("\n" + "="*60)
    print("RESULTS FOR: {}".format(sim_name))
    print("="*60)
    print("L2 Norm (RMS Error):      {:.4f} µm".format(results['l2_norm']))
    print("L-infinity Norm (Max):    {:.4f} µm".format(results['linf_norm']))
    print("\nNorm Calculation:")
    print("  L2:        sqrt(mean(error²)) over all sensors & time")
    print("  L-infinity: max(|error|) over all sensors & time")
    print("\nComparison Stats:")
    print("  Sensors compared:       {}".format(results['n_sensors']))
    print("  Time points:            {}".format(results['n_time_points']))
    print("  Time window:            [{:.4f}, {:.4f}] s".format(*results['time_window']))
    print("\nPer-Sensor RMSE (µm):")
    for i, rmse in enumerate(results['rmse_per_sensor']):
        print("  Sensor {:2d}: {:.4f}".format(i, rmse))
    print("\nPer-Sensor Max Error (µm):")
    for i, max_err in enumerate(results['max_error_per_sensor']):
        print("  Sensor {:2d}: {:.4f}".format(i, max_err))
    print("="*60 + "\n")


def plot_error_norms(results_list, output_dir='plots'):
    """
    Plot L2 and L-infinity norms vs simulation ID.
    
    Args:
        results_list: list of result dicts (must have 'sim_name', 'l2_norm', 'linf_norm')
        output_dir: directory to save plots
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Extract data
    sim_names = [r['sim_name'] for r in results_list]
    l2_norms = [r['l2_norm'] for r in results_list]
    linf_norms = [r['linf_norm'] for r in results_list]
    
    # Try to extract numeric IDs from simulation names for x-axis
    sim_ids = []
    for name in sim_names:
        try:
            # Try to extract number from end of name (e.g., "sim_001" -> 1)
            parts = name.split('_')
            numeric_part = ''.join(filter(str.isdigit, parts[-1]))
            if numeric_part:
                sim_ids.append(int(numeric_part))
            else:
                sim_ids.append(len(sim_ids))  # Fallback to index
        except:
            sim_ids.append(len(sim_ids))  # Fallback to index
    
    # --- Plot 1: L2 Norm ---
    fig1, ax1 = plt.subplots(figsize=(12, 6))
    
    ax1.plot(sim_ids, l2_norms, 'o-', color='blue', linewidth=2, markersize=6, label='L2 Norm')
    ax1.axhline(y=np.mean(l2_norms), color='red', linestyle='--', linewidth=1.5, 
                label='Mean: {:.2f} µm'.format(np.mean(l2_norms)))
    
    ax1.set_xlabel('Simulation ID', fontsize=12)
    ax1.set_ylabel('L2 Norm (RMS Error) [µm]', fontsize=12)
    ax1.set_title('L2 Norm (RMS Error) vs Simulation', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=10)
    
    # Add stats box
    stats_text = 'Min: {:.2f} µm\nMax: {:.2f} µm\nStd: {:.2f} µm'.format(
        np.min(l2_norms), np.max(l2_norms), np.std(l2_norms))
    ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes, 
             fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    l2_path = os.path.join(output_dir, 'l2_norm_comparison.png')
    fig1.savefig(l2_path, dpi=300)
    print("\nL2 norm plot saved to: {}".format(l2_path))
    plt.close(fig1)
    
    # --- Plot 2: L-infinity Norm ---
    fig2, ax2 = plt.subplots(figsize=(12, 6))
    
    ax2.plot(sim_ids, linf_norms, 'o-', color='red', linewidth=2, markersize=6, label='L-∞ Norm')
    ax2.axhline(y=np.mean(linf_norms), color='blue', linestyle='--', linewidth=1.5,
                label='Mean: {:.2f} µm'.format(np.mean(linf_norms)))
    
    ax2.set_xlabel('Simulation ID', fontsize=12)
    ax2.set_ylabel('L-∞ Norm (Max Error) [µm]', fontsize=12)
    ax2.set_title('L-infinity Norm (Max Error) vs Simulation', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=10)
    
    # Add stats box
    stats_text = 'Min: {:.2f} µm\nMax: {:.2f} µm\nStd: {:.2f} µm'.format(
        np.min(linf_norms), np.max(linf_norms), np.std(linf_norms))
    ax2.text(0.02, 0.98, stats_text, transform=ax2.transAxes,
             fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    linf_path = os.path.join(output_dir, 'linf_norm_comparison.png')
    fig2.savefig(linf_path, dpi=300)
    print("L-infinity norm plot saved to: {}".format(linf_path))
    plt.close(fig2)
    
    # --- Plot 3: Combined comparison ---
    fig3, (ax3a, ax3b) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Normalized comparison
    l2_normalized = np.array(l2_norms) / np.max(l2_norms)
    linf_normalized = np.array(linf_norms) / np.max(linf_norms)
    
    ax3a.plot(sim_ids, l2_normalized, 'o-', color='blue', linewidth=2, 
              markersize=6, label='L2 (normalized)')
    ax3a.plot(sim_ids, linf_normalized, 's-', color='red', linewidth=2,
              markersize=6, label='L-∞ (normalized)')
    ax3a.set_xlabel('Simulation ID', fontsize=12)
    ax3a.set_ylabel('Normalized Error', fontsize=12)
    ax3a.set_title('Normalized Error Comparison', fontsize=13, fontweight='bold')
    ax3a.grid(True, alpha=0.3)
    ax3a.legend(fontsize=10)
    
    # Absolute comparison
    ax3b.plot(sim_ids, l2_norms, 'o-', color='blue', linewidth=2,
              markersize=6, label='L2 Norm')
    ax3b.plot(sim_ids, linf_norms, 's-', color='red', linewidth=2,
              markersize=6, label='L-∞ Norm')
    ax3b.set_xlabel('Simulation ID', fontsize=12)
    ax3b.set_ylabel('Error [µm]', fontsize=12)
    ax3b.set_title('Absolute Error Comparison', fontsize=13, fontweight='bold')
    ax3b.grid(True, alpha=0.3)
    ax3b.legend(fontsize=10)
    
    plt.tight_layout()
    combined_path = os.path.join(output_dir, 'combined_norm_comparison.png')
    fig3.savefig(combined_path, dpi=300)
    print("Combined comparison plot saved to: {}".format(combined_path))
    plt.close(fig3)


def print_results(sim_name, results):
    """Pretty print error norm results."""
    print("\n" + "="*60)
    print("RESULTS FOR: {}".format(sim_name))
    print("="*60)
    print("L2 Norm (RMS Error):      {:.4f} um".format(results['l2_norm']))
    print("L-infinity Norm (Max):    {:.4f} um".format(results['linf_norm']))
    print("\nComparison Stats:")
    print("  Sensors compared:       {}".format(results['n_sensors']))
    print("  Time points:            {}".format(results['n_time_points']))
    print("  Time window:            [{:.4f}, {:.4f}] s".format(*results['time_window']))
    print("\nPer-Sensor RMSE (um):")
    for i, rmse in enumerate(results['rmse_per_sensor']):
        print("  Sensor {:2d}: {:.4f}".format(i, rmse))
    print("\nPer-Sensor Max Error (um):")
    for i, max_err in enumerate(results['max_error_per_sensor']):
        print("  Sensor {:2d}: {:.4f}".format(i, max_err))
    print("="*60 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description='Calculate L2 and L-infinity norms between simulation and experimental slip'
    )
    parser.add_argument('sim_name', help='Simulation name (or --batch with comma-separated list)')
    parser.add_argument('exp_csv', help='Path to experimental CSV file')
    parser.add_argument('--group', default='interface', help='Field collection group (default: interface)')
    parser.add_argument('--wdir', default='./data', help='Simulation data directory (default: ./data)')
    parser.add_argument('--t-min', type=float, help='Start time for comparison window')
    parser.add_argument('--t-max', type=float, help='End time for comparison window')
    parser.add_argument('--dense-grid', action='store_true', 
                        help='Use dense interpolated grid instead of experimental time points only')
    parser.add_argument('--batch', action='store_true', help='Batch mode: sim_name is comma-separated list')
    parser.add_argument('--output', help='Output CSV file for batch results')
    parser.add_argument('--plot-dir', default='plots', help='Directory for output plots (default: plots)')
    
    args = parser.parse_args()
    
    # Load experimental data once
    print("\n" + "="*60)
    print("Loading Experimental Data")
    print("="*60)
    exp_t, exp_data, exp_positions = load_experimental_data(args.exp_csv)
    
    # Process simulations
    if args.batch:
        sim_names = args.sim_name.split(',')
        print("\n" + "="*60)
        print("BATCH MODE: {} simulations".format(len(sim_names)))
        print("="*60)
        
        results_list = []
        for sim_name in sim_names:
            sim_name = sim_name.strip()
            print("\n" + "-"*60)
            print("Processing: {}".format(sim_name))
            print("-"*60)
            
            try:
                sim_t, sim_data, sim_positions = load_simulation_data(
                    sim_name, args.group, sensor_x=exp_positions, wdir=args.wdir
                )
                
                results = compute_error_norms(
                    sim_t, sim_data, exp_t, exp_data,
                    t_min=args.t_min, t_max=args.t_max,
                    use_exp_times=not args.dense_grid
                )
                
                results['sim_name'] = sim_name
                results_list.append(results)
                print_results(sim_name, results)
                
            except Exception as e:
                print("ERROR processing {}: {}".format(sim_name, e))
                continue
        
        # Save batch results if requested
        if args.output:
            df_rows = []
            for res in results_list:
                row = {
                    'simulation': res['sim_name'],
                    'l2_norm': res['l2_norm'],
                    'linf_norm': res['linf_norm'],
                    'n_sensors': res['n_sensors'],
                    'n_time_points': res['n_time_points']
                }
                df_rows.append(row)
            
            df = pd.DataFrame(df_rows)
            df.to_csv(args.output, index=False)
            print("\nBatch results saved to: {}".format(args.output))
            
            # Print summary
            print("\n" + "="*60)
            print("BATCH SUMMARY (sorted by L2 norm)")
            print("="*60)
            df_sorted = df.sort_values('l2_norm')
            print(df_sorted.to_string(index=False))
            print("="*60 + "\n")
        
        # Generate plots
        if results_list:
            print("\n" + "="*60)
            print("GENERATING PLOTS")
            print("="*60)
            plot_error_norms(results_list, output_dir=args.plot_dir)
    
    else:
        # Single simulation mode
        print("\n" + "="*60)
        print("Loading Simulation Data")
        print("="*60)
        
        sim_t, sim_data, sim_positions = load_simulation_data(
            args.sim_name, args.group, sensor_x=exp_positions, wdir=args.wdir
        )
        
        results = compute_error_norms(
            sim_t, sim_data, exp_t, exp_data,
            t_min=args.t_min, t_max=args.t_max,
            use_exp_times=not args.dense_grid
        )
        
        print_results(args.sim_name, results)


if __name__ == "__main__":
    main()
