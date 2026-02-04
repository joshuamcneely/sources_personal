#!/usr/bin/env python
"""
Quick diagnostic script to test waterfall plotting fixes.
Tests the experimental data loading and reports statistics.
"""
from __future__ import print_function
import sys
import os
import numpy as np

# Add ppscripts to path
sys.path.insert(0, os.path.dirname(__file__))

from waterfall_and_space_time_plot import load_experimental_data

def test_experimental_data(csv_path):
    """Test loading and processing experimental data."""
    print("="*70)
    print("TESTING EXPERIMENTAL DATA LOADING")
    print("="*70)
    print(f"\nCSV: {csv_path}\n")
    
    try:
        exp_times, exp_data_dict, exp_positions = load_experimental_data(csv_path)
        
        print("\n" + "-"*70)
        print("RESULTS:")
        print("-"*70)
        print(f"Number of sensors: {len(exp_data_dict)}")
        print(f"Number of time points: {len(exp_times)}")
        print(f"Time range: [{exp_times[0]:.6f}, {exp_times[-1]:.6f}] s")
        print(f"Time resolution (median dt): {np.median(np.diff(exp_times)):.6e} s")
        
        print("\nSensor positions:")
        for idx in sorted(exp_positions.keys()):
            print(f"  Sensor {idx}: {exp_positions[idx]:.3f} m")
        
        print("\nData statistics per sensor:")
        for idx in sorted(exp_data_dict.keys()):
            data = exp_data_dict[idx] * 1e6  # to microns
            print(f"  Sensor {idx}: min={np.min(data):.4f}, max={np.max(data):.4f}, "
                  f"mean={np.mean(data):.4f} microns")
        
        # Check for issues
        print("\n" + "-"*70)
        print("DIAGNOSTICS:")
        print("-"*70)
        
        all_data = np.array([exp_data_dict[idx] for idx in sorted(exp_data_dict.keys())])
        
        if np.allclose(all_data, 0.0):
            print("  [ERROR] All data is zero!")
        else:
            print("  [OK] Data contains non-zero values")
        
        if len(exp_data_dict) > 1:
            first = exp_data_dict[0]
            all_same = all(np.allclose(exp_data_dict[idx], first) for idx in range(1, len(exp_data_dict)))
            if all_same:
                print("  [WARNING] All sensors have identical data!")
            else:
                print("  [OK] Sensors have different data")
        
        # Check if data only varies at the end
        for idx in sorted(exp_data_dict.keys()):
            data = exp_data_dict[idx]
            # Check first half vs second half variance
            mid = len(data) // 2
            var_first = np.var(data[:mid])
            var_second = np.var(data[mid:])
            if var_first < 1e-12 and var_second > 1e-9:
                print(f"  [WARNING] Sensor {idx}: Data only varies in second half!")
        
        print("\n" + "="*70)
        print("[OK] Test completed successfully")
        print("="*70)
        
    except Exception as e:
        print(f"\n[ERROR] Failed to load data: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_waterfall_fixes.py <experimental_csv_path>")
        sys.exit(1)
    
    csv_path = sys.argv[1]
    test_experimental_data(csv_path)
