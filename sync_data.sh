#!/bin/bash
# Syncs post-processed data from HPC Ada to local ppscripts/data

# --- Configuration ---

# Remote (Ada) Settings
REMOTE_USER="pmyjm22"
REMOTE_HOST="ada"
# Path where postprocess.py stored the data on the cluster
REMOTE_DIR="/gpfs01/home/pmyjm22/uguca_project/source/postprocessing/ppscripts/data/"

# Local Settings
LOCAL_DIR="/Users/joshmcneely/sources/ppscripts/data/"

# --- Execution ---

# Ensure local directory exists
mkdir -p "$LOCAL_DIR"

echo "Syncing data from ${REMOTE_HOST}..."

# Run rsync
# -a: archive mode (preserves permissions, times, etc.)
# -v: verbose (shows details)
# -z: compress (faster transfer)
# -P: partial/progress (shows progress bar and allows resuming)
# Note: The trailing slashes (data/) ensure we merge contents, not create nested folders.
rsync -avzP "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_DIR}" "${LOCAL_DIR}"

echo "Sync complete."