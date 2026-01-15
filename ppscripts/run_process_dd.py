#!/usr/bin/env python
from __future__ import print_function, division, absolute_import
import sys
import os
import glob
import shutil
import subprocess

# --- Configuration ---
RESULTS_DIR = "/gpfs01/home/pmyjm22/uguca_project/results"
NEW_BASENAME = "dd_sim"
MAPPING_FILE = "dd_sim_mapping.txt"

def update_basename_file():
    print(f"Updating basename.txt to '{NEW_BASENAME}'")
    with open("basename.txt", "w") as f:
        f.write(NEW_BASENAME + "\n")

def ensure_source_directory():
    # Ensure RESULTS_DIR is in source_directories.txt
    if os.path.exists("source_directories.txt"):
        with open("source_directories.txt", "r") as f:
            lines = [l.strip() for l in f.readlines()]
        if RESULTS_DIR not in lines:
            print(f"Adding {RESULTS_DIR} to source_directories.txt")
            with open("source_directories.txt", "a") as f:
                f.write("\n" + RESULTS_DIR + "\n")
    else:
        with open("source_directories.txt", "w") as f:
            f.write(RESULTS_DIR + "\n")

def rename_simulations():
    """
    Renames dd_exp* files in RESULTS_DIR to dd_sim_ID* format.
    Returns the maximum ID found/created.
    """
    # 1. Check for existing dd_sim files to find start ID
    existing_sims = glob.glob(os.path.join(RESULTS_DIR, f"{NEW_BASENAME}_*.progress"))
    max_id = 0
    if existing_sims:
        ids = []
        for p in existing_sims:
            try:
                # Expected format: dd_sim_123.progress
                fname = os.path.basename(p)
                parts = fname.replace(".progress", "").split("_")
                sim_id = int(parts[-1])
                ids.append(sim_id)
            except ValueError:
                pass
        if ids:
            max_id = max(ids)
    
    print(f"Current max ID is {max_id}. Looking for new 'dd_exp' runs...")

    # 2. Find legacy dd_exp files
    # We look for .progress files as the anchor
    legacy_progress = glob.glob(os.path.join(RESULTS_DIR, "dd_exp*.progress"))
    
    if not legacy_progress:
        print("No 'dd_exp' simulations found to rename.")
        return max_id

    print(f"Found {len(legacy_progress)} 'dd_exp' simulations to rename.")
    
    # Sort to ensure deterministic order
    legacy_progress.sort()
    
    current_id = max_id
    
    # 3. Rename loop
    with open(MAPPING_FILE, "a") as f_map:
        for p_file in legacy_progress:
            current_id += 1
            
            # Extract 'dd_exp_LONG_NAME' from path/dd_exp_LONG_NAME.progress
            dir_name = os.path.dirname(p_file)
            file_name = os.path.basename(p_file)
            old_bname = file_name.replace(".progress", "")
            
            new_bname = f"{NEW_BASENAME}_{current_id}"
            
            print(f"Renaming: {old_bname} -> {new_bname}")
            f_map.write(f"{new_bname} -> {old_bname}\n")
            
            # Find all files starting with old_bname in the directory
            # We use glob with specific pattern to avoid partial matches if names overlap
            # But these names are long, overlap unlikely unless one is substring of other.
            # safe glob: old_bname + "*"
            related_files = glob.glob(os.path.join(dir_name, old_bname + "*"))
            
            for src in related_files:
                f_base = os.path.basename(src)
                if f_base.startswith(old_bname):
                    # Replace prefix
                    new_f_base = f_base.replace(old_bname, new_bname, 1)
                    dst = os.path.join(dir_name, new_f_base)
                    
                    try:
                        shutil.move(src, dst)
                    except OSError as e:
                        print(f"Error renaming {src}: {e}")
                        
    return current_id

def run_postprocess(max_id):
    if max_id == 0:
        print("No simulations to process.")
        return

    print(f"Running postprocess.py for IDs 1 to {max_id}...")
    
    # We can batch this or loop. postprocess.py takes a list?
    # sys.argv[1] can be "1,2,3-5".
    # Let's construct a range string "1-MAXID"
    range_str = f"1-{max_id}"
    
    # Command: ./postprocess.py 1-100 forced
    cmd = [sys.executable, "postprocess.py", range_str, "forced"]
    
    print(f"Executing: {' '.join(cmd)}")
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print("Error during postprocessing.")
        sys.exit(e.returncode)

def main():
    ensure_source_directory()
    max_id = rename_simulations()
    update_basename_file()
    run_postprocess(max_id)

if __name__ == "__main__":
    main()
