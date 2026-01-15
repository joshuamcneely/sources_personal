#!/usr/bin/env python
from __future__ import print_function, division, absolute_import
import sys
import os
import glob
import numpy as np
import ifasha.datamanager as idm
import ppscripts.postprocess as pp

# --- Configuration ---
WDIR = "./data"
OUTPUT_DIR = "./extracted_npz"
FIELD_NAME = 'top_disp'
# Sensor locations (meters)
SENSOR_LOCATIONS = [
    0.05, 0.25, 0.45, 0.65, 0.85, 1.05, 1.25, 1.45, 
    1.65, 1.85, 2.05, 2.25, 2.45, 2.65, 2.85, 3.05
]
MAPPING_FILE = "dd_sim_mapping.txt"

def load_mapping():
    mapping = {}
    if os.path.exists(MAPPING_FILE):
        with open(MAPPING_FILE, "r") as f:
            for line in f:
                parts = line.strip().split(" -> ")
                if len(parts) == 2:
                    mapping[parts[0]] = parts[1]
    return mapping

def extract_sim(sname, output_path):
    # Standardize name
    sname_clean = pp.sname_to_sname(sname)
    
    print(f"Extracting {sname_clean}...")
    try:
        dma = idm.DataManagerAnalysis(sname_clean, WDIR)
        
        # Robust loading of collection
        data = None
        # Try preferred names for the interface collection
        for cname in ["interface", "boundary", "contact"]:
            if cname in dma.field_collections:
                data = dma(cname)
                break
                
        if data is None:
             # Fallback
             if len(dma.field_collections) > 0:
                 first_col = list(dma.field_collections)[0]
                 print(f"  Warning: 'interface' not found. Using fallback '{first_col}'.")
                 data = dma(first_col)
             else:
                 print(f"  Skipping {sname_clean}: No field collections found.")
                 return
        
        # Identify Position Field
        if data.has_field(idm.FieldId('position', 0)):
             pos_fldid = idm.FieldId('position', 0)
        elif data.has_field(idm.FieldId('coord', 0)):
             pos_fldid = idm.FieldId('coord', 0)
        else:
             print(f"  Skipping {sname_clean}: No position field found.")
             return

        # 1. Map Sensors to Nodes (using t=0 positions)
        positions = np.array(data.get_field_at_t_index(pos_fldid, 0)[0])
        coords_x = positions.flatten()
        
        node_indices = []
        found_locs = []
        
        for target_x in SENSOR_LOCATIONS:
            # Find closest node x-coordinate
            idx = (np.abs(coords_x - target_x)).argmin()
            node_indices.append(idx)
            found_locs.append(coords_x[idx])
            
        # 2. Extract Data vs Time
        target_fld = idm.FieldId(FIELD_NAME) 
        time_fld = idm.FieldId("time")
        
        num_steps = data.get_t_index('last')
        
        times = []
        extracted_data = np.zeros((len(node_indices), num_steps))
        
        # Stride 1 for max resolution (important for earthquake nucleation)
        stride = 1 
        
        for t_idx in range(0, num_steps, stride):
             # Time
             t_val_c = data.get_field_at_t_index(time_fld, t_idx)[0]
             t_val = t_val_c[0] if isinstance(t_val_c, (np.ndarray, list)) else t_val_c
             times.append(t_val)
             
             # Data
             val_container = data.get_field_at_t_index(target_fld, t_idx)[0]
             vals = np.array(val_container).flatten()
             
             # Extract specific nodes (* 2.0 for full slip if using symmetric half-space usually, 
             # but ensure this matches your physics. 'top_disp' usually needs doubling if measuring 
             # opening/slip in symmetric vs. rigid wall, assuming symmetric here).
             extracted_data[:, len(times)-1] = vals[node_indices] * 2.0
             
        times = np.array(times)
        extracted_data = extracted_data[:, :len(times)]
        
        np.savez_compressed(output_path, time=times, data=extracted_data, locations=np.array(found_locs))
        print(f"  Saved to {output_path}")

    except Exception as e:
        print(f"  Failed {sname_clean}: {e}")


def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    mapping = load_mapping()

    # Identify available simulations in data/
    available_sims = set()
    for item in os.listdir(WDIR):
        if item.endswith("-datamanager-files"):
            available_sims.add(item.replace("-datamanager-files", ""))
        elif item.endswith(".datamanager.info"):
            available_sims.add(item.replace(".datamanager.info", ""))
            
    # Filter for relevant simulations:
    # 1. 'dd_exp' or 'dd_sim' runs
    # 2. 'original' runs referenced by the dd_exp names (e.g. dd_exp_params_vs_OriginalName)
    
    target_sims = set()
    
    # Current detected names in data/ (likely dd_sim_XX)
    dd_sims = [s for s in available_sims if "dd_exp" in s or "dd_sim" in s]
    target_sims.update(dd_sims)
    
    for dd in dd_sims:
        # Resolve original name to find comparison target
        original_sname = mapping.get(dd, dd)
        
        # Check for comparison target in name
        if "_vs_" in original_sname:
            parts = original_sname.split("_vs_")
            if len(parts) > 1:
                orig_name = parts[-1]
                # Try to fuzzy match or direct match in available sims
                if orig_name in available_sims:
                    target_sims.add(orig_name)
                # Maybe it has prefixes or suffixes?
                else:
                    # Try finding sname ending with orig_name
                    matches = [s for s in available_sims if s.endswith(orig_name)]
                    if matches:
                        target_sims.update(matches)
                        
    print(f"Found {len(dd_sims)} data-driven runs. Total extraction targets: {len(target_sims)}")

    for sname in sorted(list(target_sims)):
        # Determine output subfolder
        if "dd_exp" in sname or "dd_sim" in sname:
            subfolder = "datadriven"
        elif "gabs" in sname: 
            subfolder = "original"
        else:
            subfolder = "other"
            
        target_dir = os.path.join(OUTPUT_DIR, subfolder)
        if not os.path.exists(target_dir):
            os.makedirs(target_dir)
        
        out_name = f"sim_comparison_{sname}.npz"
        out_path = os.path.join(target_dir, out_name)
        
        if os.path.exists(out_path):
             # Optional: Check if file is valid/empty? 
             # For now, skip if exists to save time on re-runs
             print(f"Skipping {sname} (Already extracted)")
             continue
             
        extract_sim(sname, out_path)

if __name__ == "__main__":
    main()
