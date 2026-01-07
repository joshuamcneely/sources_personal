#!/usr/bin/env python
from __future__ import print_function, division, absolute_import

import sys
import os
import argparse
import numpy as np
import ifasha.datamanager as idm
import ppscripts.postprocess as pp

# --- CONFIGURATION (User to update these) ---
# List of sensor X-coordinates from the experiment (in meters)
# Example: SENSOR_LOCATIONS = [0.1, 0.2, 0.3, ...]
SENSOR_LOCATIONS = [0.05, 0.25, 0.45, 0.65, 0.85, 1.05, 1.25, 1.45, 1.65, 1.85, 2.05, 2.25, 2.45, 2.65, 2.85, 3.05]

# Defaults
DEFAULT_WDIR = "./data"
GROUP_NAME = "interface" # or 'slider_bottom', etc. depending on where data is
FIELD_NAME = "slip"      # or 'displacement'
FIELD_COMP = 0           # Component index (0 for x/tangential, 1 for y/normal)
# --------------------------------------------

def extract_and_save(sim_name, wdir):
    if not SENSOR_LOCATIONS:
        print("Error: SENSOR_LOCATIONS list is empty. Please edit the script to include target X coordinates.")
        return

    output_filename = "sim_comparison_{}.npz".format(sim_name)
    sname = pp.sname_to_sname(sim_name)
    print("Loading simulation: {}".format(sname))
    
    dma = idm.DataManagerAnalysis(sname, wdir)
    try:
        data = dma(GROUP_NAME)
    except Exception as e:
        print(f"Error loading group {GROUP_NAME}: {e}")
        sys.exit(1)

    # 1. Get Coordinates
    if data.has_field(idm.FieldId('position', 0)):
        pos_fldid = idm.FieldId('position', 0)
    elif data.has_field(idm.FieldId('coord', 0)):
        pos_fldid = idm.FieldId('coord', 0)
    else:
        print("Error: No coordinate field found.")
        sys.exit(1)

    # Get X coordinates (assuming 2D or 1D line)
    positions = np.array(data.get_field_at_t_index(pos_fldid, 0)[0])
    if positions.ndim > 1:
        coords_x = positions[:, 0]
    else:
        coords_x = positions

    # 2. Find Closest Nodes to Sensors
    node_indices = []
    found_locs = []
    
    print("Mapping sensors to simulation nodes:")
    for target_x in SENSOR_LOCATIONS:
        # Find index of closest node
        idx = (np.abs(coords_x - target_x)).argmin()
        actual_x = coords_x[idx]
        diff = abs(actual_x - target_x)
        
        print(f"  Sensor {target_x:.4f}m -> Node {idx} at {actual_x:.4f}m (diff: {diff:.2e}m)")
        node_indices.append(idx)
        found_locs.append(actual_x)

    # 3. Extract Time History
    print("Extracting time history...")
    
    # Time array
    time_fld = idm.FieldId("time")
    # Use get_t_index('last') to get the number of time steps (or end index)
    num_steps = data.get_t_index('last')
    times = []
    
    for t_idx in range(num_steps):
        t_val_c = data.get_field_at_t_index(time_fld, t_idx)[0]
        t_val = t_val_c[0] if isinstance(t_val_c, (np.ndarray, list)) else t_val_c
        times.append(t_val)
    times = np.array(times)

    # Field data
    # Shape: (Num_Sensors, Num_TimeSteps)
    field_data = np.zeros((len(node_indices), num_steps))
    
    target_fld = idm.FieldId(FIELD_NAME, FIELD_COMP)
    
    for t_idx in range(num_steps):
        val_container = data.get_field_at_t_index(target_fld, t_idx)[0]
        # val_container might be (Num_Nodes_Total, )
        vals = np.array(val_container)
        
        # Extract specific nodes
        field_data[:, t_idx] = vals[node_indices]

    print(f"Extraction complete. Shape: {field_data.shape}")
    
    # 4. Save
    np.savez(output_filename, time=times, data=field_data, locations=np.array(found_locs))
    print(f"Saved to {output_filename}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract Simulation Data for Overlay")
    parser.add_argument("sim_name", help="Name of the simulation (e.g. gabs_fine_043)")
    parser.add_argument("--wdir", default=DEFAULT_WDIR, help="Working directory containing data folder")
    
    args = parser.parse_args()
    
    extract_and_save(args.sim_name, args.wdir)
