#!/usr/bin/env python
"""
Relative Slip Visualiser (DATA Version)
------------------------------------------------
1. Computes Relative Slip (Global Zeroing).
2. Slices data to the requested Time Range.
3. AUTOMATICALLY calculates scaling to fit traces between rows.
4. Saves plots to a 'data_plots' folder.
"""

import argparse
import sys
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg') # Force non-interactive backend for HPC
import matplotlib.pyplot as plt

# --- CONFIGURATION ---
DEFAULT_T_MIN = -0.005
DEFAULT_T_MAX =  0.005
OUTPUT_FOLDER = 'data_plots'
# ---------------------

def parse_sensor_labels(columns):
    labels = []
    positions = []
    for col in columns:
        try:
            parts = col.split('_')
            pos_str = next(p for p in parts if 'm' in p and 'micro' not in p)
            val = float(pos_str.replace('m', ''))
            labels.append(f"{val:.2f} m")
            positions.append(val)
        except (StopIteration, ValueError):
            labels.append(col)
            positions.append(np.nan)
    return labels, positions

def estimate_onset(t, U, threshold_ratio=0.1):
    """Estimate onset time based on Peak Slip Rate (Max Gradient)."""
    # Mean absolute slip across all sensors
    avg_slip = np.mean(np.abs(U), axis=0)
    
    # Calculate gradient (approx slip rate)
    try:
        # Use gradient to find time of steepest rise (spike in velocity)
        grad = np.gradient(avg_slip)
        idx = np.argmax(grad)
        return t[idx]
    except Exception:
        # Fallback to threshold if gradient fails
        peak = np.max(avg_slip)
        if peak == 0: return t[0]
        idx = np.argmax(avg_slip > (peak * threshold_ratio))
        return t[idx]

def load_simulation(sim_path, t_exp_onset, time_shift=0.0, auto_align=False):
    """Load and process a single simulation file. Returns (sim_t, sim_U) or (None, None) on failure."""
    if not sim_path or not os.path.exists(sim_path):
        if sim_path:
            print(f"Warning: Simulation file not found: {sim_path}")
        return None, None
    
    print(f"Loading simulation: {sim_path}")
    try:
        sim_pkg = np.load(sim_path)
        sim_t_raw = sim_pkg['time']
        sim_data = sim_pkg['data']
        
        # Convert Simulation from Meters to Microns (1e6)
        sim_data = sim_data * 1e6

        # Zeroing relative to first time step
        sim_U = np.zeros_like(sim_data)
        for k in range(sim_data.shape[0]):
            sim_U[k, :] = sim_data[k, :] - sim_data[k, 0]

        # Auto-Alignment suggestion
        t_sim_start = estimate_onset(sim_t_raw, sim_U)
        
        # Standard Mode: Align Sim to Exp
        rec_shift = t_exp_onset - t_sim_start
        
        if auto_align:
            print(f" >> Auto-Aligning: shift {rec_shift:.5f}s (Exp start {t_exp_onset:.4f}s - Sim start {t_sim_start:.4f}s)")
            time_shift = rec_shift
        else:
            print(f" >> Alignment Hint: Exp starts ~{t_exp_onset:.5f}s, Sim starts ~{t_sim_start:.5f}s")
            print(f" >> Suggest using: --shift {rec_shift:.5f}")

        # Apply Time Shift
        sim_t = sim_t_raw + time_shift
            
        print(f"Simulation loaded: {sim_U.shape[0]} spatial points, {sim_U.shape[1]} time steps")
        return sim_t, sim_U

    except Exception as e:
        print(f"Error loading simulation npz: {e}")
        return None, None

def plot_slip_data(csv_path, t_range=None, sim_path=None, sim_time_shift=0.0, sim_scale=1.0, auto_align=False, 
                   sim_path_2=None, sim_time_shift_2=0.0, auto_align_2=False, save_suffix="", show_plot=False):
    # --- 1. Load Data ---
    if not os.path.exists(csv_path):
        sys.exit(f"Error: File not found: {csv_path}")

    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        sys.exit(f"Error reading CSV: {e}")

    time_col = df.columns[0]
    sensor_cols = df.columns[1:]
    
    t_full = df[time_col].values
    U_raw = df[sensor_cols].values.T
    num_nodes = U_raw.shape[0]

    base_name = os.path.splitext(os.path.basename(csv_path))[0]
    print(f"Processing: {base_name}")

    y_labels, sensor_positions = parse_sensor_labels(sensor_cols)
    print("Detected Sensor Positions (m):", [p for p in sensor_positions if not np.isnan(p)])

    # --- 2. Calculate Global Relative Slip ---
    U_relative_full = np.zeros_like(U_raw)
    for i in range(num_nodes):
        U_relative_full[i, :] = U_raw[i, :] - U_raw[i, 0]

    # --- 2.5 Load Simulation Data (Optional) ---
    # Estimate Exp Onset (Before shifting)
    t_exp_onset = estimate_onset(t_full, U_relative_full)

    sim_t, sim_U = load_simulation(sim_path, t_exp_onset, time_shift=sim_time_shift, auto_align=auto_align)
    sim_t_2, sim_U_2 = load_simulation(sim_path_2, t_exp_onset, time_shift=sim_time_shift_2, auto_align=auto_align_2)

    # --- 3. Time Slicing ---
    # Shift Experiment so Onset is at T=0
    # (Optional: Only if requested, currently we shift everything relative to Onset)
    t_full = t_full - t_exp_onset
    
    # If using Sim with Auto-Align, we also want to shift Sim so its onset is at T=0
    # In the code above, we shifted Sim to match Exp (which was at t_exp_onset).
    # Since we just subtracted t_exp_onset from t_full, we must subtract it from sim_t too
    # to maintain the alignment we just calculated.
    if sim_t is not None:
        sim_t = sim_t - t_exp_onset

    # Default range
    if t_range and t_range[0] is not None and t_range[0] != DEFAULT_T_MIN:
        # User supplied specific window, use it
        t_min, t_max = t_range
    else:
        # Defaults relative to aligned zero (onset)
        t_min = DEFAULT_T_MIN
        t_max = DEFAULT_T_MAX

    mask = (t_full >= t_min) & (t_full <= t_max)
    
    if np.any(mask):
        t = t_full[mask]
        U_relative = U_relative_full[:, mask]
        
        range_suffix = f"_zoom_{t_min:.4f}_{t_max:.4f}"
        title_suffix = f" ({t_min:.4f} to {t_max:.4f} s)"
    else:
        # Fallback if zoom is invalid
        print(f"Warning: No data in zoom range {t_min} to {t_max}s (after shift). Plotting full.")
        t = t_full
        U_relative = U_relative_full
        range_suffix = "_full"
        title_suffix = " (Full Duration)"

    # y_labels already parsed and unpacked above

    # --- 4. Auto-Scaling Logic ---
    # We find the max absolute value in the current time slice.
    max_val = np.max(np.abs(U_relative))
    
    if max_val == 0:
        max_val = 1.0 # Prevent division by zero if data is empty/flat
        
    # Scale so the max peak covers 80% of the gap between sensors (0.8 units)
    scaling_factor = 1.5 / max_val
    print(f"Auto-calculated scaling factor: {scaling_factor:.2e}")

    # --- 5. Plotting ---
    # Create subplots: Waterfall, Exp Heatmap, Sim1 Heatmap, Sim2 Heatmap (if available)
    num_plots = 3  # Waterfall, Exp, Sim1
    if sim_t_2 is not None:
        num_plots = 4  # Add Sim2
    
    fig, axes = plt.subplots(num_plots, 1, figsize=(10, 5*num_plots), sharex=True)
    if num_plots == 1:
        axes = [axes]  # Ensure axes is always indexable
    
    ax1 = axes[0]
    ax2 = axes[1]
    ax3 = axes[2] if num_plots >= 3 else None
    ax4 = axes[3] if num_plots >= 4 else None

    # --- NEW: ADD BIG RED TITLE ---
    fig.suptitle('DATA vs SIMULATION', fontsize=24, color='black', fontweight='bold')

    # Waterfalls (Top Plot)
    for i in range(num_nodes):
        trace = U_relative[i, :]
        
        # Apply auto-calculated factor
        visual_trace = (trace * scaling_factor) + i
        
        ax1.plot(t, visual_trace, color='red', linewidth=1.5, alpha=0.8, label='Exp' if i==0 else "")

        # Overlay Simulation 1
        if sim_U is not None and sim_t is not None:
            if i < sim_U.shape[0]:
                sim_trace = sim_U[i, :]
                sim_visual = (sim_trace * scaling_factor) + i
                ax1.plot(sim_t, sim_visual, color='blue', linestyle='--', linewidth=1.5, alpha=0.9, label='Sim 1' if i==0 else "")

        # Overlay Simulation 2
        if sim_U_2 is not None and sim_t_2 is not None:
            if i < sim_U_2.shape[0]:
                sim_trace_2 = sim_U_2[i, :]
                sim_visual_2 = (sim_trace_2 * scaling_factor) + i
                ax1.plot(sim_t_2, sim_visual_2, color='green', linestyle=':', linewidth=1.5, alpha=0.9, label='Sim 2' if i==0 else "")

    ax1.set_ylabel('Sensor Position (m)')
    ax1.set_title(f'Waterfall Comparison{title_suffix}')
    ax1.set_yticks(np.arange(num_nodes))
    ax1.set_yticklabels(y_labels)
    ax1.grid(True, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
    ax1.legend(loc='upper right')
    ax1.set_ylim(-1, num_nodes)

    # --- Heatmap 1: Experiment (Second Plot) ---
    im = ax2.imshow(U_relative, aspect='auto', origin='lower', cmap='viridis',
                    interpolation='nearest',
                    extent=[t.min(), t.max(), -0.5, num_nodes-0.5])

    ax2.set_ylabel('Sensor Position (m)')
    ax2.set_title(f'EXP Heatmap: Slip Intensity')
    ax2.set_yticks(np.arange(num_nodes))
    ax2.set_yticklabels(y_labels)
    cbar = plt.colorbar(im, ax=ax2)
    cbar.set_label(r'Slip ($\mu m$)')

    # --- Heatmap 2: Simulation 1 (Third Plot) ---
    if sim_U is not None and sim_t is not None:
        sim_mask = (sim_t >= t.min()) & (sim_t <= t.max())
        
        if np.any(sim_mask):
            sim_t_view = sim_t[sim_mask]
            sim_U_view = sim_U[:, sim_mask]
             
            im3 = ax3.imshow(sim_U_view, aspect='auto', origin='lower', cmap='viridis',
                            interpolation='bilinear',  # Finer interpolation
                            extent=[sim_t_view.min(), sim_t_view.max(), -0.5, sim_U.shape[0]-0.5])
            
            cbar3 = plt.colorbar(im3, ax=ax3)
            cbar3.set_label(r'Slip ($\mu m$)')
            ax3.set_title(f'SIM 1 Heatmap: Slip Intensity (Full Resolution)')
        else:
            ax3.text(0.5, 0.5, "No overlapping simulation data in this window", 
                     ha='center', va='center', transform=ax3.transAxes)
    else:
        if ax3:
            ax3.text(0.5, 0.5, "No Simulation 1 Data Loaded", 
                     ha='center', va='center', transform=ax3.transAxes)

    if ax3:
        ax3.set_ylabel('Spatial Location')
        ax3.set_xlabel('Time (s) [Onset at t=0]')
        ax3.set_yticks([])

    # --- Heatmap 3: Simulation 2 (Fourth Plot, if available) ---
    if ax4:
        if sim_U_2 is not None and sim_t_2 is not None:
            sim_mask_2 = (sim_t_2 >= t.min()) & (sim_t_2 <= t.max())
            
            if np.any(sim_mask_2):
                sim_t_view_2 = sim_t_2[sim_mask_2]
                sim_U_view_2 = sim_U_2[:, sim_mask_2]
                 
                im4 = ax4.imshow(sim_U_view_2, aspect='auto', origin='lower', cmap='viridis',
                                interpolation='bilinear',  # Finer interpolation
                                extent=[sim_t_view_2.min(), sim_t_view_2.max(), -0.5, sim_U_2.shape[0]-0.5])
                
                cbar4 = plt.colorbar(im4, ax=ax4)
                cbar4.set_label(r'Slip ($\mu m$)')
                ax4.set_title(f'SIM 2 Heatmap: Slip Intensity (Full Resolution)')
            else:
                ax4.text(0.5, 0.5, "No overlapping simulation data in this window", 
                         ha='center', va='center', transform=ax4.transAxes)
        else:
            ax4.text(0.5, 0.5, "No Simulation 2 Data Loaded", 
                     ha='center', va='center', transform=ax4.transAxes)

        ax4.set_ylabel('Spatial Location')
        ax4.set_xlabel('Time (s) [Onset at t=0]')
        ax4.set_yticks([])

    # Ensure all X-axes are locked to the zoom window
    ax1.set_xlim(t_min, t_max)
    ax2.set_xlim(t_min, t_max)
    ax3.set_xlim(t_min, t_max)


    # Adjust layout to make room for suptitle
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])

    # --- 6. Saving ---
    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)

    save_name = f"{base_name}{range_suffix}{save_suffix}.png"
    save_path = os.path.join(OUTPUT_FOLDER, save_name)
    
    plt.savefig(save_path, dpi=300)
    print(f"Plot saved to: {save_path}")
    
    if show_plot:
        plt.show()
    else:
        plt.close(fig)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot Relative Slip with Optional Simulation Overlay (up to 2 simulations)")
    parser.add_argument("csv_file", help="Path to experimental CSV data")
    parser.add_argument("t_min", nargs="?", type=float, default=DEFAULT_T_MIN, help="Start time (s)")
    parser.add_argument("t_max", nargs="?", type=float, default=DEFAULT_T_MAX, help="End time (s)")
    parser.add_argument("--sim", dest="sim_path", default=None, help="Path to simulation 1 .npz file for overlay")
    parser.add_argument("--shift", dest="time_shift", type=float, default=0.0, help="Time shift for simulation 1 (s)")
    parser.add_argument("--auto", dest="auto_align", action="store_true", help="Automatically align onset of slip for sim 1")
    parser.add_argument("--sim2", dest="sim_path_2", default=None, help="Path to simulation 2 .npz file for overlay")
    parser.add_argument("--shift2", dest="time_shift_2", type=float, default=0.0, help="Time shift for simulation 2 (s)")
    parser.add_argument("--auto2", dest="auto_align_2", action="store_true", help="Automatically align onset of slip for sim 2")

    args = parser.parse_args()

    # Determine relative file name for logging
    fname = os.path.basename(args.csv_file)
    print(f"Processing: {fname} | Time: {args.t_min} to {args.t_max}")
    if args.sim_path:
        print(f"Overlaying Simulation 1: {args.sim_path} (Shift: {args.time_shift}s)")
    if args.sim_path_2:
        print(f"Overlaying Simulation 2: {args.sim_path_2} (Shift: {args.time_shift_2}s)")
    if args.auto_align:
        print("Auto-alignment enabled for Sim 1")
    if args.auto_align_2:
        print("Auto-alignment enabled for Sim 2")

    plot_slip_data(args.csv_file, t_range=(args.t_min, args.t_max), 
                   sim_path=args.sim_path, sim_time_shift=args.time_shift, auto_align=args.auto_align,
                   sim_path_2=args.sim_path_2, sim_time_shift_2=args.time_shift_2, auto_align_2=args.auto_align_2)