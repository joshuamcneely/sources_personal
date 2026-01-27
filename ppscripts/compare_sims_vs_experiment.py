#!/usr/bin/env python
"""
Compare displacement time series at sensor locations for:
- Baseline (data-less) simulations
- Data-driven simulations
- Experimental data (CSV)

Produces three figures (saved under OUTPUT_DIR):
1) baseline_vs_experiment.png
2) datadriven_vs_experiment.png
3) all_vs_experiment.png

Each figure contains 16 subplots (one per sensor location) with:
- Experimental displacement (thick black)
- Simulations overlaid (thin colored)

Run from the ppscripts directory:
    python compare_sims_vs_experiment.py
"""
from __future__ import print_function, division, absolute_import

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import ifasha.datamanager as idm
import ppscripts.postprocess as pp

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
# Experimental CSV
EXP_CSV = "data_experiments/FS01-043-4MPa-RP-1_Event_1_displacement.csv"

# Simulation groups
DATA_DRIVEN_SIMS = [f"dd_exp_w_factor_study_{i:03d}" for i in range(1, 61)]
BASELINE_SIMS = [
    "gabs_fine_060",
    "gabs_fine_078",
    "gabs_fine_096",
    "gabs_fine_112",
    "gabs_fine_118",
    "gabs_fine_124",
    "gabs_fine_136",
    "gabs_fine_142",
    "gabs_fine_154",
    "gabs_fine_160",
]

# Where processed simulation data live (datamanager files)
SIM_WDIR = "./data"

# Sensors (McKlaskey layout 0.05m to 3.05m every 0.2m)
SENSOR_X = np.arange(0.05, 3.05 + 0.001, 0.2)
NUM_SENSORS = len(SENSOR_X)

# Output directory
OUTPUT_DIR = "plots_compare"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def ensure_output_dir():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print("Created directory: {}".format(OUTPUT_DIR))


def load_experimental_csv(csv_path):
    """Load experimental displacement CSV, align onset, return time and data in microns."""
    if not os.path.exists(csv_path):
        raise RuntimeError("Experimental CSV not found: {}".format(csv_path))

    df = pd.read_csv(csv_path)
    time_col = df.columns[0]
    sensor_cols = df.columns[1:]

    t_full = df[time_col].values
    U_raw = df[sensor_cols].values.T  # (n_sensors, n_times)

    # Relative slip per sensor
    U_rel = U_raw - U_raw[:, [0]]

    # Onset detection (peak slip rate)
    avg_slip = np.mean(np.abs(U_rel), axis=0)
    try:
        grad = np.gradient(avg_slip)
        onset_idx = np.argmax(grad)
        t_onset = t_full[onset_idx]
    except Exception:
        t_onset = t_full[0]

    t_aligned = t_full - t_onset

    return t_aligned, U_rel  # microns


def get_positions_and_indices(dma):
    """Return (coords_x array)."""
    data = dma("interface")
    if data.has_field(idm.FieldId("position", 0)):
        pos_fid = idm.FieldId("position", 0)
    elif data.has_field(idm.FieldId("coord", 0)):
        pos_fid = idm.FieldId("coord", 0)
    else:
        raise RuntimeError("No position/coord field in dataset")

    positions = data.get_field_at_t_index(pos_fid, 0)[0]
    positions = np.array(positions)
    if positions.ndim > 1:
        coords_x = positions[:, 0]
    else:
        coords_x = positions
    return coords_x


def extract_sim_series(sname, sensor_x=SENSOR_X, wdir=SIM_WDIR):
    """Extract top_disp at sensor positions. Returns time array (s) and matrix (sensors x time) in microns."""
    dma = idm.DataManagerAnalysis(pp.sname_to_sname(sname), wdir)
    data = dma("interface")

    coords_x = get_positions_and_indices(dma)

    # map sensors to nearest node index
    node_indices = []
    for sx in sensor_x:
        idx = int(np.argmin(np.abs(coords_x - sx)))
        node_indices.append(idx)

    time_fid = idm.FieldId("time")
    top_disp_fid = idm.FieldId("top_disp", 0)

    # time array
    time_steps = np.arange(data.get_t_index("last"))
    times = []
    for t_idx in time_steps:
        t_val_container = data.get_field_at_t_index(time_fid, t_idx)[0]
        t_val = t_val_container[0] if isinstance(t_val_container, (np.ndarray, list)) else t_val_container
        times.append(t_val)
    times = np.array(times)

    # displacement matrix
    disp = np.zeros((len(sensor_x), len(times)))
    for j, n_idx in enumerate(node_indices):
        vals = []
        for t_idx in time_steps:
            snap = data.get_field_at_t_index(top_disp_fid, t_idx)[0]
            vals.append(snap[n_idx])
        disp[j, :] = np.array(vals) * 2.0 * 1e6  # convert to total slip microns

    return times, disp


def interpolate_to_grid(target_t, src_t, src_y):
    """Interpolate src_y (sensor x time) onto target_t."""
    return np.vstack([
        np.interp(target_t, src_t, row, left=row[0], right=row[-1])
        for row in src_y
    ])


def plot_group(title, exp_t, exp_u, sim_series, fname):
    """
    sim_series: list of (label, color, matrix[sensors x time], times)
    exp_u: sensors x time (microns) aligned to exp_t
    """
    fig, axes = plt.subplots(4, 4, figsize=(16, 14), sharex=True)
    axes = axes.ravel()

    for i in range(NUM_SENSORS):
        ax = axes[i]
        ax.plot(exp_t, exp_u[i, :], color="k", linewidth=2.0, label="Experiment" if i == 0 else "")
        for label, color, t_sim, u_sim in sim_series:
            u_interp = np.interp(exp_t, t_sim, u_sim[i, :], left=u_sim[i, 0], right=u_sim[i, -1])
            ax.plot(exp_t, u_interp, color=color, alpha=0.35, linewidth=0.8, label=label if i == 0 else "")
        ax.set_title("Sensor {:.2f} m".format(SENSOR_X[i]), fontsize=10)
        if i % 4 == 0:
            ax.set_ylabel("Slip (µm)")
        if i >= 12:
            ax.set_xlabel("Time (s)")
        ax.grid(True, linestyle="--", alpha=0.5)

    # single legend
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=len(handles))

    fig.suptitle(title, fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out_path = os.path.join(OUTPUT_DIR, fname)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print("Saved {}".format(out_path))

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ensure_output_dir()

    # Load experimental
    print("Loading experimental data from {}".format(EXP_CSV))
    exp_t, exp_u_microns = load_experimental_csv(EXP_CSV)

    # Baseline sims
    baseline_series = []
    for sname in BASELINE_SIMS:
        try:
            t_sim, u_sim = extract_sim_series(sname)
            baseline_series.append(("Baseline", "#1f77b4", t_sim, u_sim))
            print("  loaded baseline {}".format(sname))
        except Exception as e:
            print("  skip baseline {}: {}".format(sname, e))

    # Data-driven sims
    dd_series = []
    for sname in DATA_DRIVEN_SIMS:
        try:
            t_sim, u_sim = extract_sim_series(sname)
            dd_series.append(("Data-driven", "#d62728", t_sim, u_sim))
            print("  loaded data-driven {}".format(sname))
        except Exception as e:
            print("  skip data-driven {}: {}".format(sname, e))

    # Plots
    if baseline_series:
        plot_group("Baseline vs Experiment", exp_t, exp_u_microns, baseline_series, "baseline_vs_experiment.png")
    if dd_series:
        plot_group("Data-Driven vs Experiment", exp_t, exp_u_microns, dd_series, "datadriven_vs_experiment.png")
    if baseline_series or dd_series:
        plot_group("All Sims vs Experiment", exp_t, exp_u_microns, baseline_series + dd_series, "all_vs_experiment.png")


if __name__ == "__main__":
    main()
