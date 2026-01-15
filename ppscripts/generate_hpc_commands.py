
import os

# The list of files provided by the user
files = [
    "dd_exp_FS01-043-10MPa-P-1_Event_3_vs_gabs_rep_2",
    "dd_exp_FS01-043-10MPa-P-1_Event_5_vs_gabs_rep_14",
    "dd_exp_FS01-043-4MPa-1_Event_6_vs_gabs_rep_2",
    "dd_exp_FS01-043-4MPa-1_Event_7_vs_gabs_rep_2",
    "dd_exp_FS01-043-4MPa-RP-1_Event_1_vs_gabs_fine_060",
    "dd_exp_FS01-043-4MPa-RP-1_Event_1_vs_gabs_fine_078",
    "dd_exp_FS01-043-4MPa-RP-1_Event_1_vs_gabs_fine_096",
    "dd_exp_FS01-043-4MPa-RP-1_Event_1_vs_gabs_fine_112",
    "dd_exp_FS01-043-4MPa-RP-1_Event_1_vs_gabs_fine_118",
    "dd_exp_FS01-043-4MPa-RP-1_Event_1_vs_gabs_fine_124",
    "dd_exp_FS01-043-4MPa-RP-1_Event_1_vs_gabs_fine_136",
    "dd_exp_FS01-043-4MPa-RP-1_Event_1_vs_gabs_fine_142",
    "dd_exp_FS01-043-4MPa-RP-1_Event_1_vs_gabs_fine_154",
    "dd_exp_FS01-043-4MPa-RP-1_Event_1_vs_gabs_fine_160"
]

# Identify distinct groups based on basename
groups = {}

print("# ------------------------------------------------------------------")
print("# 1. RENAME COMMANDS (Run these in /gpfs01/home/pmyjm22/uguca_project/results/)")
print("# ------------------------------------------------------------------")
print("cd /gpfs01/home/pmyjm22/uguca_project/results/")

for fname in files:
    parts = fname.rsplit('_', 1)
    basename = parts[0]
    sim_id_str = parts[1]
    
    # Check if we need to rename (remove leading zeros)
    sim_id = int(sim_id_str)
    new_sim_id_str = str(sim_id)
    
    if sim_id_str != new_sim_id_str:
        # Rename logic
        new_name = f"{basename}_{new_sim_id_str}"
        print(f'# Renaming {fname} to {new_name}')
        # We need to rename all extensions
        pattern = f"{fname}*"
        # Use a loop in bash to rename all related files
        print(f'for f in {fname}*; do mv "$f" "${{f/{fname}/{new_name}}}"; done')
        
        # Update our internal tracking
        groups.setdefault(basename, []).append(new_sim_id_str)
    else:
        groups.setdefault(basename, []).append(sim_id_str)

print("\n")
print("# ------------------------------------------------------------------")
print("# 2. SETUP BATCH FOLDERS (Run these in ppscripts directory)")
print("# ------------------------------------------------------------------")
print("cd /gpfs01/home/pmyjm22/uguca_project/source/postprocessing/ppscripts")

for i, (basename, ids) in enumerate(groups.items()):
    # Create a short unique directory name
    dir_name = f"batch_{i+1}"
    
    print(f"\n# Setup for {basename}")
    print(f"mkdir -p {dir_name}")
    print(f"echo '{basename}' > {dir_name}/basename.txt")
    print(f"ln -sf ../postprocess.py {dir_name}/postprocess.py")
    print(f"ln -sf ../source_directories.txt {dir_name}/source_directories.txt")
    # Also link ifasha if it's in a specific place, or assume python env handles it
    
    # Create SLURM script
    ids_joined = ",".join(ids)
    slurm_content = f"""#!/bin/bash
#SBATCH --job-name=pp_{i+1}
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=1
#SBATCH --nodes=1
#SBATCH --output=slurm-%j.out

# Ensure correct python environment
# module load python/3.x 
# source ...

python postprocess.py {ids_joined} forced
"""
    # Escape newlines for echo
    slurm_content_escaped = slurm_content.replace('\n', '\\n')
    print(f"echo -e \"{slurm_content_escaped}\" > {dir_name}/submit.slurm")

print("\n")
print("# ------------------------------------------------------------------")
print("# 3. SUBMIT JOBS")
print("# ------------------------------------------------------------------")
for i in range(len(groups)):
    dir_name = f"batch_{i+1}"
    print(f"cd {dir_name} && sbatch submit.slurm && cd ..")
