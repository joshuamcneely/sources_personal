# Helper-Repo Provenance

Entry-point scripts in this directory that were copied from the read-only
helper repo `albertini-lab/3D_uguca_sim_helper_scripts` (Gab's lab's
collection). Kept **bit-identical** to upstream to make future syncs trivial
— any local edits should be flagged here.

## Source repo
- URL:    https://github.com/albertini-lab/3D_uguca_sim_helper_scripts
- Path:   `simulation_analysis/<script>.py`
- Branch: `main`

## Files copied on 2026-05-20 from upstream commit `5a3a8f2`

| Script               | Upstream path                                       | Notes |
|----------------------|-----------------------------------------------------|-------|
| `save_rupture.py`    | `simulation_analysis/save_rupture.py`               | Entry point; wraps `ppscripts.save_ruptures.save_ruptures(...)`. Reads slip threshold `d_slip` from input `dc`. |
| `plot_3d_at_time.py` | `simulation_analysis/plot_3d_at_time.py`            | Per-time-frame visualizer; default `fldid='cohesion_1'`; iterates `['tau_max','cohesion_1']` for 2-panel rupture plots. |
| `plot_3d_xt_rpt.py`  | `simulation_analysis/plot_3d_xt_rpt.py`             | x–t space-time rupture plot; default `fldid='cohesion_1'`; also plots `tau_max` space-time. |

## Required dump fields

For these scripts to work, the simulation `.in` file must include at least:
- `cohesion_1` — primary visualization field for `plot_3d_*`
- `tau_max`    — strength field for the 2-panel comparisons
- `top_disp_1` — what `save_ruptures` thresholds against `d_slip/2` in the
  Mode I branch (`save_ruptures.py:37-40`); the rupture isochrone falls out
  of this.

## Updating from upstream

1. `cd /Users/joshmcneely/3D_uguca_sim_helper_scripts && git pull`
2. Diff each tracked script against upstream:
   `diff <upstream>/simulation_analysis/save_rupture.py ./save_rupture.py`
3. If clean, re-copy and update the commit SHA in this file's table above.
