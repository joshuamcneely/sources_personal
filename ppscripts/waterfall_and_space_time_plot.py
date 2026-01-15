#!/usr/bin/env python
from __future__ import print_function, division, absolute_import

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import ifasha.datamanager as idm
import ppscripts.postprocess as pp

# --- CONFIGURATION ---
OUTPUT_DIR = 'plots'
ROI_X_MIN = 0.0
ROI_X_MAX = 3.0  # STRICT LIMIT: Only plot sensors in this range (meters)
# ---------------------

def get_node_time_histories(sname, group, fldid, num_nodes=16, **kwargs):
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

    # Extract time array
    time_steps = np.arange(start_idx, last_idx)
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

    return time_array, node_data, node_indices, positions

def plot_dual_visualization(sname, group, fldid, scaling_factor=None, **kwargs):
    """
    Plots two subplots: Waterfall and Heatmap.
    Adds a big red 'SIMULATED' title.
    """
    num_nodes = kwargs.get('num_nodes', 16)
    
    # --- 1. Get Data (Filtered) ---
    times, data_dict, indices, all_positions = get_node_time_histories(
        sname, group, fldid, num_nodes=num_nodes, **kwargs
    )

    # Convert to Matrix
    sorted_node_indices = sorted(indices)
    U_matrix = np.zeros((len(sorted_node_indices), len(times)))
    
    y_tick_labels = []
    for i, n_idx in enumerate(sorted_node_indices):
        # DIRECT ASSIGNMENT with UNIT CONVERSION
        # 1. Multiply by 2.0: Top Disp -> Total Slip (Meters)
        # 2. Multiply by 1e6: Meters -> Microns
        U_matrix[i, :] = data_dict[n_idx] * 2.0 * 1e6
        
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
    ax1.set_title("Waterfall: Total Slip (2x {})".format(fldid.get_string()))
    
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
    ax2.set_title("Heatmap: Total Slip (2x {})".format(fldid.get_string()))
    
    ax2.set_yticks(np.arange(len(sorted_node_indices)))
    ax2.set_yticklabels(y_tick_labels)
    
    cbar = plt.colorbar(im, ax=ax2)
    cbar.set_label(r'Slip Magnitude ($\mu m$)')
    
    # --- FIX: FORCE PLAIN TEXT (NO 1e-6 OFFSET) ---
    cbar.ax.ticklabel_format(style='plain', useOffset=False)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to make room for suptitle
    return fig

if __name__ == "__main__":
    if len(sys.argv) < 4:
        sys.exit('Usage: ./waterfall_and_space_time_plot.py <sname> <group> <fldid> [scaling_factor]')

    sname = str(sys.argv[1])
    group = str(sys.argv[2])
    fldid_str = str(sys.argv[3])
    
    fldid = idm.FieldId.string_to_fieldid(fldid_str)

    scale = None
    if len(sys.argv) > 4:
        try:
            scale = float(sys.argv[4])
        except ValueError:
            print("Invalid scaling factor provided, using auto-calculation.")

    # 1. Generate Figure
    fig = plot_dual_visualization(sname, group, fldid, scaling_factor=scale)
    
    # 2. Save Logic
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print("Created directory: {}".format(OUTPUT_DIR))

    safe_fld_str = fldid_str.replace(' ', '_')
    save_name = "{}_{}.png".format(sname, safe_fld_str)
    save_path = os.path.join(OUTPUT_DIR, save_name)

    fig.savefig(save_path, dpi=300)
    print("Plot saved to: {}".format(save_path))
    
    # Close figure to prevent memory leaks in batch mode
    plt.close(fig)