#!/usr/bin/env python
from __future__ import print_function, division, absolute_import

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import ifasha.datamanager as idm
import ppscripts.postprocess as pp
import pandas as pd

# --- CONFIGURATION ---
OUTPUT_DIR = 'plots'
ROI_X_MIN = 0.0
ROI_X_MAX = 3.0  # STRICT LIMIT: Only plot sensors in this range (meters)
NUM_NODES_DEFAULT = 16  # Match experimental sensor count
# ---------------------

def select_disp_field(data):
    """Choose a displacement field id that exists in the dataset."""
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
    available = []
    try:
        available = [f.identity.get_string() for f in data.get_all_fields()]
    except Exception:
        pass
    raise RuntimeError('no displacement field found; available fields: {}'.format(', '.join(available)))


def get_node_time_histories(sname, group, fldid=None, num_nodes=16, **kwargs):
    """
    Extracts time histories for nodes strictly within the 0-3m ROI.
    Returns raw field data without normalization.
    
    Kwargs:
        target_dt (float): Target time resolution for downsampling (e.g., 1e-6 for 1 microsecond)
                          If None, uses full simulation resolution.
    """
    wdir = kwargs.get('wdir', './data')
    target_dt = kwargs.get('target_dt', None)
    sname = pp.sname_to_sname(sname)

    # 1. Load Simulation Data
    dma = idm.DataManagerAnalysis(sname, wdir)
    try:
        data = dma(group)
    except RuntimeError:
        print("Error: FieldCollection group '{}' not found in simulation '{}'.".format(group, sname))
        print("Available groups:")
        for fc in dma.get_all_field_collections():
            print(" - {}".format(fc.get_name()))
        sys.exit(1)

    # Choose displacement field if not provided
    if fldid is None:
        fldid = select_disp_field(data)

    # Probe: list available fields and sample value
    try:
        available_fields = [f.identity.get_string() for f in data.get_all_fields()]
        print("Available fields: {}".format(', '.join(available_fields)))
    except Exception:
        print("Available fields: [unable to query]")

    # 2. Determine Spatial Indices
    if data.has_field(idm.FieldId('position', 0)):
        pos_fldid = idm.FieldId('position', 0)
    elif data.has_field(idm.FieldId('coord', 0)):
        pos_fldid = idm.FieldId('coord', 0)
    else:
        raise ValueError('Dataset does not have position or coord fields.')

    # Get all spatial positions (Coordinate X)
    positions = data.get_field_at_t_index(pos_fldid, 0)[0] 
    positions = np.array(positions) 

    # Handle if positions are (N, 3) vectors or (N,) scalars
    if positions.ndim > 1:
        coords_x = positions[:, 0]
    else:
        coords_x = positions

    # --- FILTERING LOGIC ---
    # Create mask for valid range [0, 3]
    valid_mask = (coords_x >= ROI_X_MIN) & (coords_x <= ROI_X_MAX)
    valid_indices_pool = np.where(valid_mask)[0]
    
    count_valid = len(valid_indices_pool)
    print("Spatial Filtering: Found {} nodes in range [{:.2f}, {:.2f} m]".format(
        count_valid, ROI_X_MIN, ROI_X_MAX))

    if count_valid == 0:
        print("WARNING: No nodes found in ROI! Defaulting to full domain.")
        node_indices = np.linspace(0, len(positions) - 1, num_nodes, dtype=int)
    elif count_valid < num_nodes:
        # If we have fewer valid nodes than requested, take them all
        node_indices = valid_indices_pool
    else:
        # Select 'num_nodes' evenly spaced FROM THE VALID POOL
        selection_indices = np.linspace(0, count_valid - 1, num_nodes, dtype=int)
        node_indices = valid_indices_pool[selection_indices]
    
    # 3. Time setup
    start_time = kwargs.get('start_time', 0)
    end_time = kwargs.get('end_time', None)
    
    start_idx = data.get_index_of_closest_time(idm.FieldId('time'), start_time)
    if end_time is None:
        last_idx = data.get_t_index('last')
    else:
        last_idx = data.get_index_of_closest_time(idm.FieldId('time'), end_time)

    # Extract time array - with optional downsampling
    if target_dt is not None:
        # Downsample to target resolution
        time_fld = idm.FieldId("time")
        
        # Get first and last time to determine full span
        t_start_val_container = data.get_field_at_t_index(time_fld, start_idx)[0]
        t_start_val = t_start_val_container[0] if isinstance(t_start_val_container, (np.ndarray, list)) else t_start_val_container
        
        t_end_val_container = data.get_field_at_t_index(time_fld, last_idx)[0]
        t_end_val = t_end_val_container[0] if isinstance(t_end_val_container, (np.ndarray, list)) else t_end_val_container
        
        # Create target times at desired resolution
        target_times = np.arange(t_start_val, t_end_val + target_dt, target_dt)
        
        # Find closest indices for each target time
        time_steps = []
        for target_t in target_times:
            idx = data.get_index_of_closest_time(time_fld, target_t)
            if start_idx <= idx <= last_idx:
                time_steps.append(idx)
        
        time_steps = np.array(time_steps)
        print("Downsampling: Using {} time steps (target_dt={:.2e}s) out of {} available".format(
            len(time_steps), target_dt, last_idx - start_idx + 1))
    else:
        # Use full resolution (inclusive of last_idx)
        time_steps = np.arange(start_idx, last_idx + 1)
        print("Using full time resolution: {} time steps".format(len(time_steps)))

    time_array = []
    
    time_fld = idm.FieldId("time")
    for t_idx in time_steps:
        t_val_container = data.get_field_at_t_index(time_fld, t_idx)[0]
        if isinstance(t_val_container, (np.ndarray, list)):
            t_val = t_val_container[0]
        else:
            t_val = t_val_container
        time_array.append(t_val)
        
    time_array = np.array(time_array)

    # Probe: sample value from first time and first node
    try:
        sample_snapshot = data.get_field_at_t_index(fldid, time_steps[0])[0]
        sample_value = sample_snapshot[node_indices[0]]
        print("Sample value at first node/time: {:.6e}".format(sample_value))
    except Exception as e:
        print("Sample value probe failed: {}".format(e))

    # 4. Extract Field Data for Selected Nodes
    node_data = {}
    print("Extracting data for {} selected nodes...".format(len(node_indices)))

    for n_idx in node_indices:
        vals = []
        for t_idx in time_steps:
            field_snapshot = data.get_field_at_t_index(fldid, t_idx)[0]
            vals.append(field_snapshot[n_idx])
        node_data[n_idx] = np.array(vals)
    
    # DEBUG: Check if data is all zeros or identical
    all_vals = np.array([node_data[n_idx] for n_idx in node_indices])
    print("  Data shape: {}".format(all_vals.shape))
    print("  Data range: [{:.6e}, {:.6e}]".format(np.min(all_vals), np.max(all_vals)))
    if np.allclose(all_vals, 0.0):
        print("  WARNING: All data is zero!")
    if len(node_indices) > 1:
        # Check if all traces are identical
        first_trace = node_data[node_indices[0]]
        all_identical = all(np.allclose(node_data[idx], first_trace) for idx in node_indices[1:])
        if all_identical:
            print("  WARNING: All traces are identical!")

    # slip scaling factor: double only for top_disp* fields
    slip_factor = 2.0 if str(fldid.name).startswith('top_disp') else 1.0

    return time_array, node_data, node_indices, positions, slip_factor

def plot_dual_visualization(sname, group, fldid=None, scaling_factor=None, **kwargs):
    """
    Plots two subplots: Waterfall and Heatmap.
    Adds a big red 'SIMULATED' title.
    """
    num_nodes = kwargs.get('num_nodes', NUM_NODES_DEFAULT)
    kwargs_clean = dict(kwargs)
    kwargs_clean.pop('num_nodes', None)  # avoid passing num_nodes twice
    
    # --- 1. Get Data (Filtered) ---
    times, data_dict, indices, all_positions, slip_factor = get_node_time_histories(
        sname, group, fldid, num_nodes=num_nodes, **kwargs_clean
    )

    # Convert to Matrix
    sorted_node_indices = sorted(indices)
    U_matrix = np.zeros((len(sorted_node_indices), len(times)))
    
    y_tick_labels = []
    for i, n_idx in enumerate(sorted_node_indices):
        series = data_dict[n_idx]
        series = (series - series[0]) * slip_factor * 1e6
        U_matrix[i, :] = series
        
        # Get position for label
        pos_raw = all_positions[n_idx]
        if isinstance(pos_raw, (np.ndarray, list)) and len(pos_raw) > 0:
            pos_val = pos_raw[0]
        else:
            pos_val = pos_raw
            
        y_tick_labels.append("{:.2f} m".format(pos_val))
    
    # DEBUG: Check processed matrix
    print(" >> U_matrix shape: {}".format(U_matrix.shape))
    print(" >> U_matrix range after processing: [{:.4f}, {:.4f}] microns".format(np.min(U_matrix), np.max(U_matrix)))
    if np.allclose(U_matrix, 0.0):
        print(" >> WARNING: U_matrix is all zeros after processing!")
    if len(sorted_node_indices) > 1:
        all_same = all(np.allclose(U_matrix[i, :], U_matrix[0, :]) for i in range(1, len(sorted_node_indices)))
        if all_same:
            print(" >> WARNING: All rows in U_matrix are identical!")

    # --- 2. Auto-Scaling Logic (Visual Only) ---
    if scaling_factor is None:
        max_val = np.max(np.abs(U_matrix))
        if max_val == 0: max_val = 1.0
        
        # Factor 1.5 ensures traces are tall enough to see details
        # U_matrix is in microns, so max_val might be e.g., 50.0
        scaling_factor = 1.5 / max_val
        print("Auto-calculated scaling factor: {:.2e}".format(scaling_factor))
    else:
        print("Using provided scaling factor: {:.2e}".format(scaling_factor))

    # --- 3. Plotting ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # DEBUG: Verify values
    print(" >> Plotting Data Range: {:.4f} to {:.4f} microns".format(np.min(U_matrix), np.max(U_matrix)))

    # --- NEW: ADD BIG RED TITLE ---
    fig.suptitle('SIMULATED', fontsize=30, color='red', fontweight='bold')

    # --- SUBPLOT 1: WATERFALL ---
    for i in range(len(sorted_node_indices)):
        trace = U_matrix[i, :]
        # Visual offset only; data integrity preserved
        visual_trace = (trace * scaling_factor) + i
        
        # Red traces, no fill
        ax1.plot(times, visual_trace, color='red', linewidth=0.8)

    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Spatial Position (m)')
    fld_name = fldid.get_string() if fldid is not None else 'auto-detected field'
    ax1.set_title("Waterfall: Slip ({})".format(fld_name))
    
    ax1.set_yticks(np.arange(len(sorted_node_indices)))
    ax1.set_yticklabels(y_tick_labels)
    ax1.grid(True, color='gray', linestyle='--', linewidth=0.5, alpha=0.7)
    
    # Fix Y-limits to match the number of nodes
    ax1.set_ylim(-1, len(sorted_node_indices))

    # --- SUBPLOT 2: HEATMAP ---
    im = ax2.imshow(U_matrix, aspect='auto', origin='lower', cmap='viridis',
                    interpolation='nearest',
                    extent=[times.min(), times.max(), -0.5, len(sorted_node_indices)-0.5])
    
    ax2.set_xlabel('Time (s)')
    ax2.set_title("Heatmap: Slip ({})".format(fld_name))
    
    ax2.set_yticks(np.arange(len(sorted_node_indices)))
    ax2.set_yticklabels(y_tick_labels)
    
    cbar = plt.colorbar(im, ax=ax2)
    cbar.set_label(r'Slip Magnitude ($\mu m$)')
    
    # --- FIX: FORCE PLAIN TEXT (NO 1e-6 OFFSET) ---
    cbar.ax.ticklabel_format(style='plain', useOffset=False)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to make room for suptitle
    return fig

def plot_velocity_heatmap(sname, group, **kwargs):
    """
    Plots a heatmap of velocity data (top_velo field).
    """
    num_nodes = kwargs.get('num_nodes', NUM_NODES_DEFAULT)
    kwargs_clean = dict(kwargs)
    kwargs_clean.pop('num_nodes', None)
    
    # --- 1. Get Velocity Data (Filtered) ---
    velo_fldid = idm.FieldId('top_velo', 0)
    times, data_dict, indices, all_positions, _ = get_node_time_histories(
        sname, group, velo_fldid, num_nodes=num_nodes, **kwargs_clean
    )

    # Convert to Matrix
    sorted_node_indices = sorted(indices)
    V_matrix = np.zeros((len(sorted_node_indices), len(times)))
    
    y_tick_labels = []
    for i, n_idx in enumerate(sorted_node_indices):
        # Velocity in m/s, convert to mm/s for better visualization
        V_matrix[i, :] = data_dict[n_idx] * 1e3
        
        # Get position for label
        pos_raw = all_positions[n_idx]
        if isinstance(pos_raw, (np.ndarray, list)) and len(pos_raw) > 0:
            pos_val = pos_raw[0]
        else:
            pos_val = pos_raw
            
        y_tick_labels.append("{:.2f} m".format(pos_val))

    # --- 2. Plotting ---
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # DEBUG: Verify values
    print(" >> Velocity Data Range: {:.4f} to {:.4f} mm/s".format(np.min(V_matrix), np.max(V_matrix)))

    # --- ADD BIG RED TITLE ---
    fig.suptitle('SIMULATED', fontsize=30, color='red', fontweight='bold')

    # --- HEATMAP ---
    im = ax.imshow(V_matrix, aspect='auto', origin='lower', cmap='hot',
                    interpolation='nearest',
                    extent=[times.min(), times.max(), -0.5, len(sorted_node_indices)-0.5])
    
    ax.set_xlabel('Time (s)', fontsize=12)
    ax.set_ylabel('Spatial Position (m)', fontsize=12)
    ax.set_title("Velocity Heatmap (top_velo)", fontsize=14)
    
    ax.set_yticks(np.arange(len(sorted_node_indices)))
    ax.set_yticklabels(y_tick_labels)
    
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(r'Velocity (mm/s)', fontsize=12)
    cbar.ax.ticklabel_format(style='plain', useOffset=False)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    return fig

def load_experimental_data(csv_path, nb_nodes=2048, duration=0.015, nb_steps=None):
    """
    Load and align experimental data from CSV.
    Returns: times, data_dict (indexed by sensor index 0-15), positions
    """
    print(f"Loading experimental data from: {csv_path}")
    
    # Read CSV
    df = pd.read_csv(csv_path)
    time_col = df.columns[0]
    sensor_cols = df.columns[1:]
    
    t_full = df[time_col].values
    U_raw = df[sensor_cols].values.T  # Shape: (n_sensors, n_times)
    
    print(f"  Raw data shape: {U_raw.shape} (sensors x times)")
    
    # Calculate relative slip
    U_relative_full = np.zeros_like(U_raw)
    for i in range(U_raw.shape[0]):
        U_relative_full[i, :] = U_raw[i, :] - U_raw[i, 0]
    
    # Onset detection (peak slip rate)
    avg_slip = np.mean(np.abs(U_relative_full), axis=0)
    try:
        grad = np.gradient(avg_slip)
        onset_idx = np.argmax(grad)
        t_onset = t_full[onset_idx]
    except:
        t_onset = t_full[0]
    
    print(f"  Experimental onset detected at t = {t_onset:.6f} s")
    
    # Time alignment
    t_aligned = t_full - t_onset
    
    # Window extraction
    if nb_steps is None:
        dx = 6.0 / nb_nodes  # Default domain length
        cs_approx = 3000.0
        stable_dt = dx / cs_approx
        time_step_factor = 0.1
        dt_sim = time_step_factor * stable_dt
        nb_steps = int(np.ceil(duration / dt_sim))
    
    t_start = -duration / 2
    t_end = duration / 2
    window_mask = (t_aligned >= t_start) & (t_aligned <= t_end)
    
    t_windowed = t_aligned[window_mask]
    U_windowed = U_relative_full[:, window_mask]
    
    print(f"  Extracted {len(t_windowed)} experimental time points")
    print(f"  Windowed data shape: {U_windowed.shape} (sensors x times)")
    
    # Sensor positions (assuming McKlaskey setup: 0.05 to 3.05 m, 0.2 m spacing)
    sensors_x = np.arange(0.05, 3.05 + 0.01, 0.2)
    n_sensors = min(len(sensors_x), U_windowed.shape[0])
    
    # Map to simulation time grid
    t_sim = t_windowed - t_windowed[0]
    
    exp_data_dict = {}
    exp_positions = {}
    
    # Use sequential sensor indices (0 to n_sensors-1)
    for i in range(n_sensors):
        slip_microns = U_windowed[i, :]
        slip_meters = slip_microns * 1e-6  # Convert to meters
        
        # Store with original time grid (don't interpolate here)
        exp_data_dict[i] = slip_meters
        exp_positions[i] = sensors_x[i]
    
    print(f"  Loaded {len(exp_data_dict)} sensors at positions: {[exp_positions[k] for k in sorted(exp_positions.keys())]}")
    
    return t_sim, exp_data_dict, exp_positions

def plot_combined_waterfall(
    sname,
    group,
    fldid=None,
    exp_data_dict=None,
    exp_positions=None,
    exp_times=None,
    scaling_factor=None,
    **kwargs
):
    """
    Plots waterfall with both simulation and experimental data overlaid.
    """
    num_nodes = kwargs.get('num_nodes', NUM_NODES_DEFAULT)
    kwargs_clean = dict(kwargs)
    kwargs_clean.pop('num_nodes', None)
    
    # --- 1. Get Simulation Data ---
    times, data_dict, indices, all_positions, slip_factor = get_node_time_histories(
        sname, group, fldid, num_nodes=num_nodes, **kwargs_clean
    )

    sorted_node_indices = sorted(indices)
    U_sim_matrix = np.zeros((len(sorted_node_indices), len(times)))
    
    y_tick_labels = []
    for i, n_idx in enumerate(sorted_node_indices):
        series = data_dict[n_idx]
        series = (series - series[0]) * slip_factor * 1e6
        U_sim_matrix[i, :] = series
        
        pos_raw = all_positions[n_idx]
        if isinstance(pos_raw, (np.ndarray, list)) and len(pos_raw) > 0:
            pos_val = pos_raw[0]
        else:
            pos_val = pos_raw
            
        y_tick_labels.append("{:.2f} m".format(pos_val))

    # --- 2. Auto-Scaling ---
    if scaling_factor is None:
        max_val = np.max(np.abs(U_sim_matrix))
        if max_val == 0: max_val = 1.0
        scaling_factor = 1.5 / max_val
        print("Auto-calculated scaling factor: {:.2e}".format(scaling_factor))

    # --- 3. Plotting ---
    fig, ax = plt.subplots(figsize=(14, 10))
    fig.suptitle('SIMULATED vs EXPERIMENTAL', fontsize=30, color='red', fontweight='bold')

    # Plot simulation data (red)
    for i in range(len(sorted_node_indices)):
        trace_sim = U_sim_matrix[i, :]
        visual_trace_sim = (trace_sim * scaling_factor) + i
        ax.plot(times, visual_trace_sim, color='red', linewidth=1.2, label='Simulation' if i == 0 else "")

    # Plot experimental data (blue) if provided
    if exp_data_dict is not None and exp_positions is not None and exp_times is not None:
        for sensor_idx in sorted(exp_data_dict.keys()):
            exp_data = exp_data_dict[sensor_idx]
            pos = exp_positions[sensor_idx]
            
            # Find closest node index for y-position
            closest_idx = None
            for i, n_idx in enumerate(sorted_node_indices):
                node_pos_raw = all_positions[n_idx]
                if isinstance(node_pos_raw, (np.ndarray, list)) and len(node_pos_raw) > 0:
                    node_pos = node_pos_raw[0]
                else:
                    node_pos = node_pos_raw
                
                if abs(node_pos - pos) < 0.1:  # Within 0.1m
                    closest_idx = i
                    break
            
            if closest_idx is not None:
                # Interpolate experimental data to simulation times
                exp_series = np.interp(times, exp_times, exp_data, left=exp_data[0], right=exp_data[-1])
                exp_microns = exp_series * 1e6
                visual_trace_exp = (exp_microns * scaling_factor) + closest_idx
                ax.plot(times, visual_trace_exp, color='blue', linewidth=1.2, linestyle='--', 
                        label='Experimental' if sensor_idx == 0 else "")

    ax.set_xlabel('Time (s)', fontsize=12)
    ax.set_ylabel('Spatial Position (m)', fontsize=12)
    ax.set_title("Waterfall: Simulation (red) vs Experimental (blue dashed)", fontsize=14)
    
    ax.set_yticks(np.arange(len(sorted_node_indices)))
    ax.set_yticklabels(y_tick_labels)
    ax.grid(True, color='gray', linestyle='--', linewidth=0.5, alpha=0.7)
    ax.set_ylim(-1, len(sorted_node_indices))
    ax.legend(loc='upper right', fontsize=12)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    return fig

def plot_experimental_heatmap(exp_data_dict, exp_positions, exp_times, sname, **kwargs):
    """
    Plots heatmap of experimental data only.
    """
    if not exp_data_dict or not exp_positions:
        print("No experimental data to plot.")
        return None
    
    # Sort by position
    sorted_sensors = sorted(exp_positions.items(), key=lambda x: x[1])
    sorted_indices = [s[0] for s in sorted_sensors]
    sorted_positions = [s[1] for s in sorted_sensors]
    
    print(f"  Experimental heatmap: {len(sorted_indices)} sensors")
    
    # Build matrix
    n_sensors = len(sorted_indices)
    n_times = len(exp_data_dict[sorted_indices[0]])
    U_exp_matrix = np.zeros((n_sensors, n_times))
    
    y_tick_labels = []
    for i, sensor_idx in enumerate(sorted_indices):
        U_exp_matrix[i, :] = exp_data_dict[sensor_idx] * 1e6  # to microns
        y_tick_labels.append("{:.2f} m".format(sorted_positions[i]))
    
    # --- Plotting ---
    fig, ax = plt.subplots(figsize=(12, 8))
    fig.suptitle('EXPERIMENTAL DATA', fontsize=30, color='blue', fontweight='bold')
    
    print(" >> Experimental Data Range: {:.4f} to {:.4f} microns".format(np.min(U_exp_matrix), np.max(U_exp_matrix)))
    
    # Use time extent if available
    if exp_times is not None and len(exp_times) > 0:
        time_extent = [exp_times[0], exp_times[-1], -0.5, n_sensors-0.5]
    else:
        time_extent = [0, n_times-1, -0.5, n_sensors-0.5]
    
    im = ax.imshow(U_exp_matrix, aspect='auto', origin='lower', cmap='viridis',
                    interpolation='nearest',
                    extent=time_extent)
    
    ax.set_xlabel('Time (s)' if exp_times is not None else 'Time Step', fontsize=12)
    ax.set_ylabel('Spatial Position (m)', fontsize=12)
    ax.set_title("Experimental Heatmap: {} sensors".format(n_sensors), fontsize=14)
    
    ax.set_yticks(np.arange(n_sensors))
    ax.set_yticklabels(y_tick_labels)
    
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(r'Slip Magnitude ($\mu m$)', fontsize=12)
    cbar.ax.ticklabel_format(style='plain', useOffset=False)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    return fig

def plot_multi_overlay_waterfall(sname_list, group, fldid=None, exp_csv_list=None, scaling_factor=None, **kwargs):
    """
    Plots waterfall with multiple simulations and/or multiple experimental datasets overlaid.
    
    Args:
        sname_list: list of simulation names
        exp_csv_list: list of experimental CSV paths (optional)
    """
    num_nodes = kwargs.get('num_nodes', NUM_NODES_DEFAULT)
    kwargs_clean = dict(kwargs)
    kwargs_clean.pop('num_nodes', None)
    
    # Color cycles
    sim_colors = ['red', 'orange', 'purple', 'brown', 'pink']
    exp_colors = ['blue', 'cyan', 'green', 'lime', 'teal']
    
    # --- 1. Load all simulations ---
    sim_data_list = []
    all_positions = None
    sorted_node_indices = None
    times = None
    
    for i, sname in enumerate(sname_list):
        print("Loading simulation {}/{}: {}".format(i+1, len(sname_list), sname))
        sim_times, data_dict, indices, positions, slip_factor = get_node_time_histories(
            sname, group, fldid, num_nodes=num_nodes, **kwargs_clean
        )
        
        # Use first simulation for spatial grid reference
        if all_positions is None:
            all_positions = positions
            sorted_node_indices = sorted(indices)
            times = sim_times
        
        # Convert to matrix
        U_matrix = np.zeros((len(sorted_node_indices), len(sim_times)))
        for j, n_idx in enumerate(sorted_node_indices):
            if n_idx in data_dict:
                series = data_dict[n_idx]
                series = (series - series[0]) * slip_factor * 1e6
                U_matrix[j, :] = series
        
        sim_data_list.append((sname, sim_times, U_matrix, sim_colors[i % len(sim_colors)]))
    
    # --- 2. Load all experimental datasets (if provided) ---
    exp_data_list = []
    if exp_csv_list:
        for i, csv_path in enumerate(exp_csv_list):
            print("Loading experimental data {}/{}: {}".format(i+1, len(exp_csv_list), csv_path))
            try:
                _, exp_dict, exp_pos = load_experimental_data(csv_path)
                exp_data_list.append((csv_path, exp_dict, exp_pos, exp_colors[i % len(exp_colors)]))
            except Exception as e:
                print("Warning: Could not load {}: {}".format(csv_path, e))
    
    # --- 3. Auto-Scaling ---
    if scaling_factor is None:
        max_vals = [np.max(np.abs(d[2])) for d in sim_data_list]
        max_val = max(max_vals) if max_vals else 1.0
        scaling_factor = 1.5 / max_val
        print("Auto-calculated scaling factor: {:.2e}".format(scaling_factor))
    
    # --- 4. Create y-axis labels ---
    y_tick_labels = []
    for n_idx in sorted_node_indices:
        pos_raw = all_positions[n_idx]
        if isinstance(pos_raw, (np.ndarray, list)) and len(pos_raw) > 0:
            pos_val = pos_raw[0]
        else:
            pos_val = pos_raw
        y_tick_labels.append("{:.2f} m".format(pos_val))
    
    # --- 5. Plotting ---
    fig, ax = plt.subplots(figsize=(16, 10))
    fig.suptitle('MULTI-OVERLAY: Simulations vs Experiments', fontsize=24, color='black', fontweight='bold')
    
    # Plot all simulations
    for sname, sim_t, U_matrix, color in sim_data_list:
        for i in range(len(sorted_node_indices)):
            trace = U_matrix[i, :]
            visual_trace = (trace * scaling_factor) + i
            label = 'Sim: {}'.format(sname) if i == 0 else ""
            ax.plot(sim_t, visual_trace, color=color, linewidth=1.0, alpha=0.7, label=label)
    
    # Plot all experimental datasets
    for csv_name, exp_dict, exp_pos, color in exp_data_list:
        base_name = os.path.basename(csv_name)
        for sensor_idx, exp_data in exp_dict.items():
            pos = exp_pos[sensor_idx]
            # Find closest node for y-position
            closest_idx = None
            for i, n_idx in enumerate(sorted_node_indices):
                node_pos_raw = all_positions[n_idx]
                if isinstance(node_pos_raw, (np.ndarray, list)) and len(node_pos_raw) > 0:
                    node_pos = node_pos_raw[0]
                else:
                    node_pos = node_pos_raw
                
                if abs(node_pos - pos) < 0.1:
                    closest_idx = i
                    break
            
            if closest_idx is not None:
                exp_microns = exp_data * 1e6
                visual_trace = (exp_microns * scaling_factor) + closest_idx
                label = 'Exp: {}'.format(base_name) if sensor_idx == list(exp_dict.keys())[0] else ""
                ax.plot(times, visual_trace, color=color, linewidth=1.2, linestyle='--', 
                        alpha=0.8, label=label)
    
    ax.set_xlabel('Time (s)', fontsize=12)
    ax.set_ylabel('Spatial Position (m)', fontsize=12)
    ax.set_title("Multi-Overlay Waterfall", fontsize=14)
    
    ax.set_yticks(np.arange(len(sorted_node_indices)))
    ax.set_yticklabels(y_tick_labels)
    ax.grid(True, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
    ax.set_ylim(-1, len(sorted_node_indices))
    ax.legend(loc='upper right', fontsize=10, ncol=2)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    return fig

if __name__ == "__main__":
    if len(sys.argv) < 3:
        sys.exit('''Usage: ./waterfall_and_space_time_plot.py <sname> <group> [fldid] [exp_csv_path] [scaling_factor]
        
    OR for multi-overlay:
        ./waterfall_and_space_time_plot.py --multi <group> [fldid] --sims sim1,sim2,sim3 --exps exp1.csv,exp2.csv
    
    Arguments:
        sname           : Simulation name (e.g., dd_no_nucleation_data_only)
        group           : Field collection group (e.g., interface)
        fldid           : Field ID (e.g., top_disp, 0) or 'auto' to auto-detect [default: auto]
        exp_csv_path    : Path to experimental CSV file (optional, for overlay)
        scaling_factor  : Manual scaling factor for visualization (optional)
    
    Multi-overlay mode:
        --multi         : Enable multi-overlay mode
        --sims          : Comma-separated list of simulation names
        --exps          : Comma-separated list of experimental CSV paths (optional)
    
    Examples:
        python waterfall_and_space_time_plot.py dd_no_nucleation_data_only interface auto
        python waterfall_and_space_time_plot.py dd_no_nucleation_data_only interface auto exp_data.csv
        python waterfall_and_space_time_plot.py --multi interface auto --sims sim1,sim2,sim3 --exps exp1.csv,exp2.csv
        ''')

    # Check for multi-overlay mode
    if sys.argv[1] == '--multi':
        # Multi-overlay mode
        group = str(sys.argv[2])
        fldid_str = str(sys.argv[3]) if len(sys.argv) > 3 else 'auto'
        fldid = None if fldid_str.lower() == 'auto' else idm.FieldId.string_to_fieldid(fldid_str)
        
        # Parse --sims and --exps
        sim_list = []
        exp_list = []
        
        i = 4
        while i < len(sys.argv):
            if sys.argv[i] == '--sims' and i+1 < len(sys.argv):
                sim_list = sys.argv[i+1].split(',')
                i += 2
            elif sys.argv[i] == '--exps' and i+1 < len(sys.argv):
                exp_list = sys.argv[i+1].split(',')
                i += 2
            else:
                i += 1
        
        if not sim_list:
            sys.exit("Error: --multi mode requires --sims argument")
        
        print("\n=== MULTI-OVERLAY MODE ===")
        print("Simulations: {}".format(', '.join(sim_list)))
        if exp_list:
            print("Experiments: {}".format(', '.join(exp_list)))
        
        if not os.path.exists(OUTPUT_DIR):
            os.makedirs(OUTPUT_DIR)
        
        # Detect experimental time resolution if experiments provided
        exp_dt = None
        if exp_list:
            try:
                df = pd.read_csv(exp_list[0])
                exp_times = df[df.columns[0]].values
                if len(exp_times) > 1:
                    exp_dt = np.median(np.diff(exp_times))
                    print("[OK] Detected experimental time resolution: {:.2e} s".format(exp_dt))
            except:
                pass
        
        kwargs_overlay = {'num_nodes': NUM_NODES_DEFAULT}
        if exp_dt:
            kwargs_overlay['target_dt'] = exp_dt
        
        fig_multi = plot_multi_overlay_waterfall(
            sim_list, group, fldid, 
            exp_csv_list=exp_list if exp_list else None,
            **kwargs_overlay
        )
        
        save_name = "multi_overlay_{}_sims.png".format(len(sim_list))
        save_path = os.path.join(OUTPUT_DIR, save_name)
        fig_multi.savefig(save_path, dpi=300)
        print("\nMulti-overlay plot saved to: {}".format(save_path))
        plt.close(fig_multi)
        
        sys.exit(0)
    
    # Standard single-simulation mode
    sname = str(sys.argv[1])
    group = str(sys.argv[2])
    fldid_str = str(sys.argv[3]) if len(sys.argv) > 3 else 'auto'
    
    fldid = None if fldid_str.lower() == 'auto' else idm.FieldId.string_to_fieldid(fldid_str)

    exp_csv_path = None
    scale = None
    
    # Parse remaining arguments (exp_csv_path and/or scaling_factor)
    if len(sys.argv) > 4:
        arg4 = sys.argv[4]
        # Check if it's a file path or a number
        if os.path.exists(arg4):
            exp_csv_path = arg4
            if len(sys.argv) > 5:
                try:
                    scale = float(sys.argv[5])
                except ValueError:
                    pass
        else:
            try:
                scale = float(arg4)
            except ValueError:
                exp_csv_path = arg4 if arg4 else None

    # Load experimental data if provided
    exp_data_dict = None
    exp_positions = None
    exp_dt = None  # Experimental time resolution
    
    if exp_csv_path:
        try:
            exp_times, exp_data_dict, exp_positions = load_experimental_data(exp_csv_path)
            print("[OK] Experimental data loaded successfully from: {}".format(exp_csv_path))
            
            # Detect experimental time resolution for downsampling
            df = pd.read_csv(exp_csv_path)
            exp_times_raw = df[df.columns[0]].values
            if len(exp_times_raw) > 1:
                exp_dt = np.median(np.diff(exp_times_raw))
                print("[OK] Detected experimental time resolution: {:.2e} s ({:.2f} us)".format(
                    exp_dt, exp_dt * 1e6))
        except Exception as e:
            print("[WARNING] Could not load experimental data: {}".format(e))
    else:
        print("[INFO] No experimental CSV provided (optional)")

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print("Created directory: {}".format(OUTPUT_DIR))

    # 1. Generate Simulation Waterfall + Heatmap (dual visualization) - FULL RESOLUTION
    print("\n=== Generating Simulation Plots (Full Resolution) ===")
    fig = plot_dual_visualization(sname, group, fldid, scaling_factor=scale, num_nodes=NUM_NODES_DEFAULT)
    
    safe_fld_str = fldid_str.replace(' ', '_')
    save_name = "{}_{}.png".format(sname, safe_fld_str)
    save_path = os.path.join(OUTPUT_DIR, save_name)

    fig.savefig(save_path, dpi=300)
    print("Simulation plot saved to: {}".format(save_path))
    plt.close(fig)
    
    # 2. Generate Velocity Heatmap - FULL RESOLUTION (optional if field exists)
    print("\n=== Generating Velocity Heatmap (Full Resolution) ===")
    try:
        fig_velo = plot_velocity_heatmap(sname, group, num_nodes=NUM_NODES_DEFAULT)
        
        velo_save_name = "{}_top_velo_heatmap.png".format(sname)
        velo_save_path = os.path.join(OUTPUT_DIR, velo_save_name)
        
        fig_velo.savefig(velo_save_path, dpi=300)
        print("Velocity heatmap saved to: {}".format(velo_save_path))
        plt.close(fig_velo)
    except (KeyError, RuntimeError) as e:
        print("[INFO] Velocity field not available, skipping velocity heatmap: {}".format(e))
    
    # 3. Generate Combined Waterfall (Simulation + Experimental) - DOWNSAMPLED for efficiency
    if exp_data_dict:
        print("\n=== Generating Combined Waterfall (Downsampled to Exp Resolution) ===")
        
        # Use experimental time resolution for comparison plot
        kwargs_downsampled = {'num_nodes': NUM_NODES_DEFAULT}
        if exp_dt is not None:
            kwargs_downsampled['target_dt'] = exp_dt
        
        fig_combined = plot_combined_waterfall(
            sname,
            group,
            fldid,
            exp_data_dict,
            exp_positions,
            exp_times=exp_times,
            scaling_factor=scale,
            **kwargs_downsampled
        )
        
        combined_save_name = "{}_combined_waterfall.png".format(sname)
        combined_save_path = os.path.join(OUTPUT_DIR, combined_save_name)
        
        fig_combined.savefig(combined_save_path, dpi=300)
        print("Combined waterfall saved to: {}".format(combined_save_path))
        plt.close(fig_combined)
    
    # 4. Generate Experimental Heatmap - NATIVE RESOLUTION (as measured)
    if exp_data_dict:
        print("\n=== Generating Experimental Heatmap (Native Resolution) ===")
        fig_exp = plot_experimental_heatmap(exp_data_dict, exp_positions, exp_times, sname)
        
        exp_save_name = "{}_experimental_heatmap.png".format(sname)
        exp_save_path = os.path.join(OUTPUT_DIR, exp_save_name)
        
        fig_exp.savefig(exp_save_path, dpi=300)
        print("Experimental heatmap saved to: {}".format(exp_save_path))
        plt.close(fig_exp)
    
    print("\n=== All plots generated successfully ===")