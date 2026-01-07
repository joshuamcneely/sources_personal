#!/bin/bash
# Script to run extract_sim_for_overlay.py on all gabs_fine simulations

for d in data/gabs_fine_*.datamanager.info; do
    if [ -e "$d" ]; then
        # Extract filename from path
        filename=$(basename "$d")
        # Remove extension to get simulation name
        sim_name="${filename%.datamanager.info}"
        
        echo "---------------------------------------------------"
        echo "Processing Simulation: $sim_name"
        
        # Run the python extraction script
        python extract_sim_for_overlay.py "$sim_name"
    fi
done

echo "Done! All .npz files should be generated."
