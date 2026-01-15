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
FIELD_ID = idm.FieldId('top_disp', 0) # 0 = Tangential, 1 = Normal usually.
FIELD_NAME = 'slip'
NB_X_POINTS = 1000 # Spatial resolution
NB_T_POINTS = 1000 # Temporal resolution (or use -1 for all?)
# Using -1 for time might be huge. But we need enough resolution for the comp.
# The experiments are usually 1-4 MHz. 
# If we re-interpolate in the comparison script, maybe 1000-2000 points is enough?
# Let's try to get a decent number.
MATCH_SIMS = ["gabs_fine", "dd_exp"]

def extract_sim(sname, output_path):
    print(f"Extracting {sname}...")
    try:
        dma = idm.DataManagerAnalysis(sname, WDIR)
        data = dma('interface')
        
        # Position Field
        if data.has_field(idm.FieldId('position', 0)):
             pos_fldid = idm.FieldId('position', 0)
        elif data.has_field(idm.FieldId('coord', 0)):
             pos_fldid = idm.FieldId('coord', 0)
        else:
             print(f"  Skipping {sname}: No position field found.")
             return

        # Time range
        start = 0
        last = data.get_t_index('last')
        
        # We want high time resolution.
        # If we just grab everything, it might be slow but accurate.
        # Let's try a stride.
        stride = 1
        if last > 5000:
             stride = last // 2000 # Aim for ~2000 time steps
        
        t_indices = np.arange(start, last, stride)
        
        # Space range (Full interface)
        # We handle slicing manually to ensure we get X coords
        nb_elements = data.get_field_shape(pos_fldid)[1]
        x_stride = max(1, nb_elements // NB_X_POINTS)
        x_indices = np.arange(0, nb_elements, x_stride)

        # Extract
        # get_sliced_x_sliced_t_plot returns X, T, Z
        # X: (space,) or (space, time)? usually (space,)
        # T: (time,) usually (time, 1) or similar
        # Z: (time, space) or (space, time) check documentation or test.
        # Usually standard plotting tools return Z as (time, space) for pcolormesh?
        # Let's check typical usage. extract_data_to_npz_file uses it.
        
        X, T, Z = data.get_sliced_x_sliced_t_plot(
            pos_fldid,
            x_indices,
            idm.FieldId("time"),
            t_indices,
            FIELD_ID
        )
        
        # Check shapes
        # X should be locations
        # T should be time
        # Z should be data
        
        # Make sure T is 1D
        T = T.flatten()
        X = X.flatten()
        
        # Ensure Z is (space, time) for our comparison script
        # The comparison script expects sim_data[s_idx, :] -> trace over time
        # So (n_locs, n_time)
        
        if Z.shape[0] == len(T) and Z.shape[1] == len(X):
             # It is (time, space), transpose it
             Z = Z.T
        elif Z.shape[0] == len(X) and Z.shape[1] == len(T):
             # It is (space, time), good
             pass
        else:
             print(f"  Warning: Unexpected shape Z={Z.shape}, X={X.shape}, T={T.shape}")
             # fallback guess
             if Z.shape[0] == len(T): Z = Z.T

        # Save
        np.savez_compressed(output_path, time=T, data=Z, locations=X)
        print(f"  Saved to {output_path}")

    except Exception as e:
        print(f"  Failed {sname}: {e}")

def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # Find directories
    all_items = os.listdir(WDIR)
    sim_dirs = [d for d in all_items if d.endswith("-datamanager-files") or d.endswith(".datamanager.info")]
    
    # Unique sim names
    processed_names = set()
    
    for item in sorted(sim_dirs):
        # Extract sname
        if item.endswith("-datamanager-files"):
            sname = item.replace("-datamanager-files", "")
        elif item.endswith(".datamanager.info"):
            sname = item.replace(".datamanager.info", "")
        else:
            continue
            
        if sname in processed_names: continue
        
        # Filter
        is_match = any(m in sname for m in MATCH_SIMS)
        if not is_match: continue
        
        # Determine output subfolder
        if "gabs_fine" in sname:
            subfolder = "original"
        elif "dd_exp" in sname:
            subfolder = "datadriven"
        else:
            subfolder = "other"
            
        target_dir = os.path.join(OUTPUT_DIR, subfolder)
        if not os.path.exists(target_dir):
            os.makedirs(target_dir)
        
        out_name = f"sim_comparison_{sname}.npz"
        out_path = os.path.join(target_dir, out_name)
        
        if os.path.exists(out_path):
             # Skip if exists? Or overwrite? 
             # print(f"Skipping {sname}, already exists.")
             # continue
             pass
             
        processed_names.add(sname)
        extract_sim(sname, out_path)

if __name__ == "__main__":

    main()

