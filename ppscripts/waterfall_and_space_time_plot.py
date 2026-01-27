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
NUM_NODES_DEFAULT = 48  # Increased for finer resolution
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
    """
    wdir = kwargs.get('wdir', './data')
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

    # Extract time array (inclusive of last_idx)
    time_steps = np.arange(start_idx, last_idx + 1)
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

    # 4. Extract Field Data for Selected Nodes
    node_data = {}
    print("Extracting data for {} selected nodes...".format(len(node_indices)))

    for n_idx in node_indices:
        vals = []
        for t_idx in time_steps:
            field_snapshot = data.get_field_at_t_index(fldid, t_idx)[0]
            vals.append(field_snapshot[n_idx])
        node_data[n_idx] = np.array(vals)

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
    ax1.set_title("Waterfall: Slip ({})".format(fldid.get_string()))
    
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
    ax2.set_title("Heatmap: Slip ({})".format(fldid.get_string()))
    
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
    Returns: times, data_dict (indexed by spatial position), positions
    """
    print(f"Loading experimental data from: {csv_path}")
    
    # Read CSV
    df = pd.read_csv(csv_path)
    time_col = df.columns[0]
    sensor_cols = df.columns[1:]
    
    t_full = df[time_col].values
    U_raw = df[sensor_cols].values.T  # Shape: (n_sensors, n_times)
    
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
    
    print(f"Experimental onset detected at t = {t_onset:.6f} s")
    
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
    
    print(f"Extracted {len(t_windowed)} experimental time points")
    
    # Sensor positions (assuming McKlaskey setup: 0.05 to 3.05 m, 0.2 m spacing)
    sensors_x = np.arange(0.05, 3.05 + 0.01, 0.2)
    dx = 6.0 / nb_nodes
    sensor_nodes = np.round(sensors_x / dx).astype(int)
    valid_mask = sensor_nodes < nb_nodes
    valid_sensor_nodes = sensor_nodes[valid_mask]
    valid_positions = sensors_x[valid_mask]
    
    # Map to simulation time grid
    t_sim = t_windowed - t_windowed[0]
    dt_sim = duration / nb_steps
    
    exp_data_dict = {}
    exp_positions = {}
    
    for sensor_idx, pos in zip(valid_sensor_nodes, valid_positions):
        slip_microns = U_windowed[sensor_idx - valid_sensor_nodes[0], :] if sensor_idx - valid_sensor_nodes[0] < U_windowed.shape[0] else np.zeros(len(t_windowed))
        slip_meters = slip_microns * 1e-6  # Convert to meters (experimental already total slip)
        
        # Interpolate to simulation timesteps
        sim_times = np.arange(nb_steps + 1) * dt_sim
        exp_interp = np.interp(sim_times, t_sim, slip_meters, left=0.0, right=slip_meters[-1])
        exp_data_dict[sensor_idx] = exp_interp
        exp_positions[sensor_idx] = pos
    
    return t_sim, exp_data_dict, exp_positions

def plot_combined_waterfall(sname, group, fldid=None, exp_data_dict=None, exp_positions=None, scaling_factor=None, **kwargs):
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
    if exp_data_dict is not None and exp_positions is not None:
        for sensor_idx, exp_data in exp_data_dict.items():
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
                exp_microns = exp_data * 1e6
                visual_trace_exp = (exp_microns * scaling_factor) + closest_idx
                ax.plot(times, visual_trace_exp, color='blue', linewidth=1.2, linestyle='--', 
                        label='Experimental' if sensor_idx == list(exp_data_dict.keys())[0] else "")

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

def plot_experimental_heatmap(exp_data_dict, exp_positions, sname, **kwargs):
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
    
    im = ax.imshow(U_exp_matrix, aspect='auto', origin='lower', cmap='viridis',
                    interpolation='nearest',
                    extent=[0, n_times, -0.5, n_sensors-0.5])
    
    ax.set_xlabel('Time Step', fontsize=12)
    ax.set_ylabel('Spatial Position (m)', fontsize=12)
    ax.set_title("Experimental Heatmap", fontsize=14)
    
    ax.set_yticks(np.arange(n_sensors))
    ax.set_yticklabels(y_tick_labels)
    
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(r'Slip Magnitude ($\mu m$)', fontsize=12)
    cbar.ax.ticklabel_format(style='plain', useOffset=False)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    return fig

if __name__ == "__main__":
    if len(sys.argv) < 4:
        sys.exit('Usage: ./waterfall_and_space_time_plot.py <sname> <group> <fldid> [exp_csv_path] [scaling_factor]')

    sname = str(sys.argv[1])
    group = str(sys.argv[2])
    fldid_str = str(sys.argv[3])
    
    fldid = None if fldid_str.lower() == 'auto' else idm.FieldId.string_to_fieldid(fldid_str)

    exp_csv_path = None
    scale = None
    
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
                pass

    # Load experimental data if provided
    exp_data_dict = None
    exp_positions = None
    if exp_csv_path:
        try:
            _, exp_data_dict, exp_positions = load_experimental_data(exp_csv_path)
        except Exception as e:
            print(f"Warning: Could not load experimental data: {e}")

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print("Created directory: {}".format(OUTPUT_DIR))

    # 1. Generate Simulation Waterfall + Heatmap (dual visualization)
    print("\n=== Generating Simulation Plots ===")
    fig = plot_dual_visualization(sname, group, fldid, scaling_factor=scale, num_nodes=NUM_NODES_DEFAULT)
    
    safe_fld_str = fldid_str.replace(' ', '_')
    save_name = "{}_{}.png".format(sname, safe_fld_str)
    save_path = os.path.join(OUTPUT_DIR, save_name)

    fig.savefig(save_path, dpi=300)
    print("Simulation plot saved to: {}".format(save_path))
    plt.close(fig)
    
    # 2. Generate Velocity Heatmap
    print("\n=== Generating Velocity Heatmap ===")
    fig_velo = plot_velocity_heatmap(sname, group, num_nodes=NUM_NODES_DEFAULT)
    
    velo_save_name = "{}_top_velo_heatmap.png".format(sname)
    velo_save_path = os.path.join(OUTPUT_DIR, velo_save_name)
    
    fig_velo.savefig(velo_save_path, dpi=300)
    print("Velocity heatmap saved to: {}".format(velo_save_path))
    plt.close(fig_velo)
    
    # 3. Generate Combined Waterfall (Simulation + Experimental)
    if exp_data_dict:
        print("\n=== Generating Combined Waterfall ===")
        fig_combined = plot_combined_waterfall(sname, group, fldid, exp_data_dict, exp_positions, 
                                              scaling_factor=scale, num_nodes=NUM_NODES_DEFAULT)
        
        combined_save_name = "{}_combined_waterfall.png".format(sname)
        combined_save_path = os.path.join(OUTPUT_DIR, combined_save_name)
        
        fig_combined.savefig(combined_save_path, dpi=300)
        print("Combined waterfall saved to: {}".format(combined_save_path))
        plt.close(fig_combined)
    
    # 4. Generate Experimental Heatmap
    if exp_data_dict:
        print("\n=== Generating Experimental Heatmap ===")
        fig_exp = plot_experimental_heatmap(exp_data_dict, exp_positions, sname)
        
        exp_save_name = "{}_experimental_heatmap.png".format(sname)
        exp_save_path = os.path.join(OUTPUT_DIR, exp_save_name)
        
        fig_exp.savefig(exp_save_path, dpi=300)
        print("Experimental heatmap saved to: {}".format(exp_save_path))
        plt.close(fig_exp)
    
    print("\n=== All plots generated successfully ===")