#!/usr/bin/env python
from __future__ import print_function, division, absolute_import
import sys
import os
import glob
import subprocess
import numpy as np
import ifasha.datamanager as idm
import ppscripts.postprocess as pp

# --- Configuration ---
WDIR = "./data"
OUTPUT_DIR = "./extracted_npz"
FIELD_NAME = 'top_disp'
# Using specific sensor locations from the experiment setup
SENSOR_LOCATIONS = [
    0.05, 0.25, 0.45, 0.65, 0.85, 1.05, 1.25, 1.45, 
    1.65, 1.85, 2.05, 2.25, 2.45, 2.65, 2.85, 3.05
]
MATCH_SIMS = ["gabs_fine", "dd_exp"]

def ensure_postprocessed(sname):
    """
    Checks if a simulation is properly post-processed in data/.
    If not, runs postprocess_simple.py.
    """
    dm_dir = os.path.join(WDIR, f"{sname}-datamanager-files")
    fc_dir = os.path.join(dm_dir, "interface-fieldcollection-files")
    
    needs_pp = False
    if not os.path.exists(dm_dir) or not os.path.exists(fc_dir) or not os.listdir(fc_dir):
        needs_pp = True
        
    if needs_pp:
        print(f"  [Auto-Fix] Post-processing missing for {sname}. Running postprocess_simple.py...")
        
        # We need to find the source directory for this sname
        source_dirs = []
        if os.path.exists("source_directories.txt"):
            with open("source_directories.txt", "r") as f:
                source_dirs = [line.strip() for line in f if line.strip() and not line.startswith("#")]
        
        # Explicit look in known results dir
        extra_result_path = "/gpfs01/home/pmyjm22/uguca_project/results"
        if extra_result_path not in source_dirs and os.path.exists(extra_result_path):
             source_dirs.append(extra_result_path)
             
        found_path = None
        
        # Search strategy matching the user's structure
        for path in source_dirs:
            # 1. Direct folder match (likely for dd_exp)
            chk_path = os.path.join(path, sname)
            if os.path.isdir(chk_path):
                found_path = chk_path
                breaking = True
                break
            
            # 2. File match sname.progress (flat structure)
            if os.path.exists(os.path.join(path, sname + ".progress")):
                found_path = os.path.join(path, sname) # postprocess_simple splits this
                break
                
        if not found_path:
            print(f"    Error: Could not find source for {sname} in {len(source_dirs)} scan paths. Skipping.")
            return

        # Call existing script
        try:
            cmd = [sys.executable, "postprocess_simple.py", found_path]
            # Use input 'p' to pass if prompted (the script asks input if exists, but we are fixing missing ones so it should proceed)
            # But just in case, input='p\n'
            subprocess.run(cmd, check=True, input=b"p\n", stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            print(f"    Success: {sname} processed.")
        except subprocess.CalledProcessError as e:
            print(f"    Failed to run postprocess_simple.py: {e.stderr.decode('utf-8')}")

def extract_sim(sname, output_path):
    # sname convention handling
    sname = pp.sname_to_sname(sname)
    
    # Check and Auto-Fix
    ensure_postprocessed(sname)
    
    print(f"Extracting {sname}...")
    try:
        dma = idm.DataManagerAnalysis(sname, WDIR)
        
        # Robust loading of collection
        data = None
        # Try preferred names
        for cname in ["interface", "boundary", "contact"]:
            try:
                # Check if it exists in dma.field_collections to avoid raising Exception blindly
                # dma.field_collections is a set of strings
                if cname in dma.field_collections:
                    data = dma(cname)
                    print(f"  Using collection: {cname}")
                    break
            except Exception:
                pass
                
        if data is None:
             # Fallback: take the first available one
             if len(dma.field_collections) > 0:
                 first_col = list(dma.field_collections)[0]
                 print(f"  Warning: 'interface' not found. Fallback to '{first_col}'.")
                 data = dma(first_col)
             else:
                 print(f"  Skipping {sname}: No field collections found.")
                 return
        
        # Position Field
        if data.has_field(idm.FieldId('position', 0)):
             pos_fldid = idm.FieldId('position', 0)
        elif data.has_field(idm.FieldId('coord', 0)):
             pos_fldid = idm.FieldId('coord', 0)
        else:
             print(f"  Skipping {sname}: No position field found.")
             return

        # 1. Map Sensors to Nodes
        # Get all spatial coordinates (only need to do this once at t=0)
        # Assuming field shape is (1, NumNodes) or (NumNodes,)
        positions = np.array(data.get_field_at_t_index(pos_fldid, 0)[0])
        # Flatten if necessary
        coords_x = positions.flatten()
        
        node_indices = []
        found_locs = []
        
        for target_x in SENSOR_LOCATIONS:
            # Find closest node
            idx = (np.abs(coords_x - target_x)).argmin()
            
            # Optional: Warning if too far? 
            # if abs(coords_x[idx] - target_x) > 0.02: print("Warning: poor match")
            
            node_indices.append(idx)
            found_locs.append(coords_x[idx])
            
        # 2. Extract Data vs Time (High Resolution)
        # We grab all time steps because we need high fidelity for comparison
        target_fld = idm.FieldId(FIELD_NAME) 
        
        # Get total time steps
        num_steps = data.get_t_index('last')
        
        # Time Array
        # We can extract time at all steps
        time_fld = idm.FieldId("time")
        # NOTE: getting field at t index for time is inefficient if we do it one by one
        # but the library might not have a "get all times" function exposed easily here.
        # Let's hope it's fast enough or check if there's a better way.
        # Actually most efficient way in this library pattern is looping.
        
        times = []
        # Pre-allocate data array: (NumSensors, NumTimeSteps)
        extracted_data = np.zeros((len(node_indices), num_steps))
        
        # Checking strides? If it's too slow, maybe stride=2
        stride = 1 
        # For 40000 steps it might take a while.
        # But for accurate comparison we usually want 1-5 microsecond resolution.
        
        for t_idx in range(0, num_steps, stride):
             # Time
             t_val_c = data.get_field_at_t_index(time_fld, t_idx)[0]
             # Unpack if list
             t_val = t_val_c[0] if isinstance(t_val_c, (np.ndarray, list)) else t_val_c
             times.append(t_val)
             
             # Data
             val_container = data.get_field_at_t_index(target_fld, t_idx)[0]
             vals = np.array(val_container).flatten()
             
             # Extract specific nodes * 2.0 (for full slip)
             extracted_data[:, len(times)-1] = vals[node_indices] * 2.0
             
        times = np.array(times)
        # Ensure data is trimmed if we strided
        extracted_data = extracted_data[:, :len(times)]
        
        # Save
        # Compare script expects: sim_data[s_idx, :] -> trace over time
        # We have (NumSensors, NumTimeSteps), which matches.
        np.savez_compressed(output_path, time=times, data=extracted_data, locations=np.array(found_locs))
        print(f"  Saved to {output_path}")

    except Exception as e:
        print(f"  Failed {sname}: {e}")


def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # Find directories
    all_items = os.listdir(WDIR)
    
    # 1. Identify all available simulation names
    # Strategy: 
    # A) Look in data/ for existing stuff (Original sims likely here)
    # B) Look in source_directories.txt for date-driven stuff (dd_exp) that might not be processed yet
    
    available_sims = set()
    
    # A) Check data/
    for item in os.listdir(WDIR):
        if item.endswith("-datamanager-files"):
            available_sims.add(item.replace("-datamanager-files", ""))
        elif item.endswith(".datamanager.info"):
            available_sims.add(item.replace(".datamanager.info", ""))

    # B) Check source directories for dd_exp runs
    if os.path.exists("source_directories.txt"):
        with open("source_directories.txt") as f:
            sdirs = [l.strip() for l in f if l.strip() and not l.startswith("#")]
    else:
        sdirs = []
        
    # Explicitly add user's results dir if known and missing (matches user req)
    user_results = "/gpfs01/home/pmyjm22/uguca_project/results"
    if user_results not in sdirs and os.path.isdir(user_results):
        sdirs.append(user_results)

    print(f"Scanning source directories for 'dd_exp' candidates...")
    found_dd = 0
    for sdir in sdirs:
        if not os.path.exists(sdir): continue
        # Look for folders or .progress files starting with dd_exp
        # 1. Folders (if raw format)
        candidates = [d for d in os.listdir(sdir) if d.startswith("dd_exp")]
        for c in candidates:
            # If it's a directory
            if os.path.isdir(os.path.join(sdir, c)):
                 # Usually the dir name is the sname? 
                 # Or if it ends with -interface-DataFiles (which generates inside results)? 
                 # User 'ls' showed folders like: dd_exp_...-interface-DataFiles
                 # But also raw .progress files.
                 
                 # Case 1: The folder is the sname (raw output folder style) - specific to Akantu/WI?
                 # Case 2: The folder is ALREADY a datamanager folder (like -datamanager-files)
                 
                 clean_name = c
                 if clean_name.endswith("-datamanager-files"):
                     clean_name = clean_name.replace("-datamanager-files", "")
                 elif clean_name.endswith("-interface-DataFiles"):
                     # This structure implies it might be partially processed or raw output style
                     clean_name = clean_name.replace("-interface-DataFiles", "")
                 
                 if "dd_exp" in clean_name:
                     available_sims.add(clean_name)
                     found_dd += 1

            # If it's a file (like .progress)
            elif c.endswith(".progress"):
                 clean_name = c.replace(".progress", "")
                 available_sims.add(clean_name)
                 found_dd += 1

    print(f"Found {len(available_sims)} total candidate simulations (found {found_dd} new dd candidates).")

    # 2. Identify Data Driven simulations and their corresponding Original simulations
    dd_sims = [s for s in available_sims if "dd_exp" in s]
    target_sims = set(dd_sims)
    
    print(f"Found {len(dd_sims)} data-driven simulations.")
    
    for dd in dd_sims:
        # Expected format: "dd_exp_..._vs_ORIGINAL_NAME"
        if "_vs_" in dd:
            parts = dd.split("_vs_")
            if len(parts) > 1:
                orig_name = parts[-1]
                # Check if this orig_name exists in available_sims
                if orig_name in available_sims:
                    target_sims.add(orig_name)
                else:
                    # Sometimes suffixes or slight naming differences occur?
                    # For now just warn
                    print(f"  Note: Original sim '{orig_name}' referenced by '{dd}' not found in directory.")

    print(f"Total target simulations to extract: {len(target_sims)}")

    # 3. Extract
    for sname in sorted(list(target_sims)):

        # Determine output subfolder
        if "dd_exp" in sname:
            subfolder = "datadriven"
        elif "gabs" in sname: # Covers gabs_fine, gabs_rep, etc.
            subfolder = "original"
        else:
            subfolder = "other"
            
        target_dir = os.path.join(OUTPUT_DIR, subfolder)
        if not os.path.exists(target_dir):
            os.makedirs(target_dir)
        
        out_name = f"sim_comparison_{sname}.npz"
        out_path = os.path.join(target_dir, out_name)
        
        if os.path.exists(out_path):
             print(f"Skipping {sname}, already exists at {out_path}")
             continue
             
        extract_sim(sname, out_path)

if __name__ == "__main__":

    main()

