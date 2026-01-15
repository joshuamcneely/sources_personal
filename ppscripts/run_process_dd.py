#!/usr/bin/env python
from __future__ import print_function, division, absolute_import
import sys
import os
import glob
import subprocess

# --- Configuration ---
WDIR = "./data"
# Explicit result path on HPC
EXTRA_RESULT_PATH = "/gpfs01/home/pmyjm22/uguca_project/results"

def find_source_candidates():
    candidates = []
    
    # 1. Check source_directories.txt
    if os.path.exists("source_directories.txt"):
        with open("source_directories.txt", "r") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and os.path.exists(line):
                    candidates.append(line)
    
    # 2. Check EXTRA_RESULT_PATH
    if os.path.exists(EXTRA_RESULT_PATH) and EXTRA_RESULT_PATH not in candidates:
        candidates.append(EXTRA_RESULT_PATH)
        
    return candidates

def find_dd_sims_to_process(search_paths):
    """
    Scans search paths for folders starting with 'dd_exp'.
    Returns a dictionary {sname: source_path}
    """
    to_process = {}
    
    print(f"Scanning for 'dd_exp' simulations in {len(search_paths)} locations...")
    
    for path in search_paths:
        try:
            items = os.listdir(path)
        except OSError:
            continue
            
        for item in items:
            full_path = os.path.join(path, item)
            
            # Logic to identify a simulation source folder
            if item.startswith("dd_exp"):
                # Case 1: It's a directory
                if os.path.isdir(full_path):
                    sname = item
                    # cleanup suffix if present from previous simple postprocess checks
                    if sname.endswith("-interface-DataFiles"):
                        sname = sname.replace("-interface-DataFiles", "")
                    
                    to_process[sname] = full_path
                
                # Case 2: It's a .progress file
                elif item.endswith(".progress"):
                    sname = item.replace(".progress", "")
                    # The source path for postprocess_simple is the folder containing the .progress usually,
                    # or the .progress file itself (postprocess_simple handles both often, but let's pass the folder)
                    to_process[sname] = path
                    
    print(f"Found {len(to_process)} unique 'dd_exp' candidates.")
    return to_process

def is_processed(sname):
    """Checks if valid structure exists in data/"""
    dm_dir = os.path.join(WDIR, f"{sname}-datamanager-files")
    fc_dir = os.path.join(dm_dir, "interface-fieldcollection-files")
    
    # We consider it processed if the FC folder exists and is not empty
    if os.path.exists(dm_dir) and os.path.exists(fc_dir) and len(os.listdir(fc_dir)) > 0:
        return True
    return False

def run_postprocess(sname, source_path):
    print(f"Processing {sname}...")
    
    # Construct the path argument for postprocess_simple.py
    # If source_path is the parent folder (from search), we append sname?
    # Actually, postprocess_simple usually takes the simulation DIRECTORY.
    
    # If we found it via .progress in 'results/', the simulation dir is 'results/sname' (if it exists)
    # OR 'results/' if it's a flat structure.
    # Let's try to be specific.
    
    target_path = source_path
    if not source_path.endswith(sname) and os.path.isdir(os.path.join(source_path, sname)):
         target_path = os.path.join(source_path, sname)
         
    cmd = [sys.executable, "postprocess_simple.py", target_path]
    
    try:
        # 'input=b"p\n"' automatically answers "p" if the script prompts for interaction
        subprocess.run(cmd, check=True, input=b"p\n", stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print(f"  [OK] {sname}")
    except subprocess.CalledProcessError as e:
        print(f"  [FAILED] {sname}")
        print(f"  Error: {e.stderr.decode('utf-8')}")

def main():
    if not os.path.exists(WDIR):
        os.makedirs(WDIR)
        
    search_paths = find_source_candidates()
    sims = find_dd_sims_to_process(search_paths)
    
    count_processed = 0
    count_skipped = 0
    
    sorted_snames = sorted(sims.keys())
    
    for sname in sorted_snames:
        if is_processed(sname):
            count_skipped += 1
            # print(f"Skipping {sname} (Already processed)")
            continue
            
        source = sims[sname]
        run_postprocess(sname, source)
        count_processed += 1
        
    print("-" * 40)
    print(f"Summary: Processed {count_processed}, Skipped {count_skipped} (already done).")

if __name__ == "__main__":
    main()
