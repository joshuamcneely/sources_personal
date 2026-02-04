#!/usr/bin/env python
"""
Merge error results from batch processing into comprehensive CSV files.

Run after all error calculation jobs complete.
"""
import pandas as pd
import glob
import os

print("="*60)
print("MERGING ERROR RESULTS")
print("="*60)

error_dir = "plots/error_plots"

# Merge dd_study results (001-060)
print("\n1. Merging dd_exp_w_factor_study results...")
dd_files = [
    f"{error_dir}/dd_study_01-20_error_results.csv",
    f"{error_dir}/dd_study_21-40_error_results.csv",
    f"{error_dir}/dd_study_41-60_error_results.csv"
]

dd_dfs = []
for f in dd_files:
    if os.path.exists(f):
        df = pd.read_csv(f)
        dd_dfs.append(df)
        print(f"   [OK] Loaded {f}: {len(df)} simulations")
    else:
        print(f"   [MISSING] {f}")

if dd_dfs:
    dd_merged = pd.concat(dd_dfs, ignore_index=True)
    dd_merged = dd_merged.sort_values('simulation')
    output_file = f"{error_dir}/dd_study_ALL_error_results.csv"
    dd_merged.to_csv(output_file, index=False)
    print(f"\n   [OK] Merged {len(dd_merged)} simulations")
    print(f"   [OK] Saved to: {output_file}")
    
    # Summary statistics
    print(f"\n   Summary Statistics:")
    print(f"     L2 Norm:  min={dd_merged['l2_norm'].min():.2f}, max={dd_merged['l2_norm'].max():.2f}, mean={dd_merged['l2_norm'].mean():.2f}")
    print(f"     L-inf:    min={dd_merged['linf_norm'].min():.2f}, max={dd_merged['linf_norm'].max():.2f}, mean={dd_merged['linf_norm'].mean():.2f}")
    
    # Best simulation
    best_idx = dd_merged['l2_norm'].idxmin()
    best_sim = dd_merged.loc[best_idx, 'simulation']
    best_l2 = dd_merged.loc[best_idx, 'l2_norm']
    print(f"\n   Best simulation: {best_sim} (L2={best_l2:.2f} microns)")
else:
    print("   [ERROR] No dd_study results found")

# Check gabs_fine results
print("\n2. Checking gabs_fine results...")
gabs_file = f"{error_dir}/gabs_fine_error_results.csv"
if os.path.exists(gabs_file):
    gabs_df = pd.read_csv(gabs_file)
    print(f"   [OK] Loaded {gabs_file}: {len(gabs_df)} simulations")
    print(f"\n   Summary Statistics:")
    print(f"     L2 Norm:  min={gabs_df['l2_norm'].min():.2f}, max={gabs_df['l2_norm'].max():.2f}, mean={gabs_df['l2_norm'].mean():.2f}")
    print(f"     L-inf:    min={gabs_df['linf_norm'].min():.2f}, max={gabs_df['linf_norm'].max():.2f}, mean={gabs_df['linf_norm'].mean():.2f}")
    
    best_idx = gabs_df['l2_norm'].idxmin()
    best_sim = gabs_df.loc[best_idx, 'simulation']
    best_l2 = gabs_df.loc[best_idx, 'l2_norm']
    print(f"\n   Best simulation: {best_sim} (L2={best_l2:.2f} microns)")
else:
    print(f"   [MISSING] {gabs_file}")

print("\n" + "="*60)
print("MERGE COMPLETE")
print("="*60)
