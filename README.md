# Updated README for `xbeach-processing`

## Purpose of this Directory

The **`xbeach-processing`** repository contains a collection of Python scripts and utilities for processing, analysing, and visualising output from **XBeach** when used to sequentially remove buildings due to hurricane damage. An accompanying paper is being prepared. This code provides:

- Helper functions for handling paths and common plotting tasks.
- A suite of specialised modules (each in its own sub‑directory) that generate figures, animations, statistical summaries, and confusion matrices from XBeach model runs.
- A central driver script (`main.py`) that imports all of these modules and offers convenient entry‑points for the most common workflows.
- Configuration via `paths.txt` where the user specifies locations of model output, forcing data, building footprints, etc.

The top‑level `README.md` (shown below) gives a quick start guide, while each sub‑directory contains its own README that explains the specific functionality of that module.

## Files in the Root Directory

| File | Description |
|------|-------------|
| `main.py` | Example driver script that imports every processing class.  Users can uncomment the desired section(s) to generate specific plots, animations, or statistics. |
| `paths.txt` | Configuration file where the user sets paths to the XBeach model run, forcing file, building GeoJSON, damage CSV, and output directory. |
| `updated_README.md` | **(This file)** – a consolidated description of the repository, its purpose, and a summary of the top‑level files. |

## Overview of Sub‑directories

Each sub‑directory implements a focused analysis or visualisation task and includes its own `README.md` with detailed usage information.  Below is a brief catalogue:

- `compare_ds_w_stats` – Compare damage states with wave statistics; produces violin plots and optional decision‑tree visualisations.
- `compare_forcing_output` – Compare model forcing input to model output.
- `confusion_matrix` – Generate confusion‑matrix plots for damage‑state classification.
- `extract_stats_point` – Extract time‑series statistics at specific points.
- `helpers` – Shared helper functions (path handling, figure saving, etc.).
- `hotstart_make_animation` – Create animations for hot‑start runs.
- `hotstart_removed_bldgs` – Visualise removed buildings in hot‑start scenarios.
- `julia_scripts` – (Placeholder) scripts written in Julia.
- `make_animation` – General animation creation from XBeach fields.
- `plot_bldg_dmg` – Plot building damage maps.
- `plot_current_quiver` – Quiver plots of current vectors.
- `plot_forcing` – Plot forcing time series (e.g., water elevation, wave height).
- `plot_grid` – Visualise model grid depth and domain.
- `plot_high_water_marks` – Scatter plot of high‑water‑mark locations.
- `plot_input` – Visualise input files.
- `plot_output_point` – Time‑series plots at specific output points.
- `plot_output_transect` – Transect plots of model output.
- `plot_stats_v_dcoast` – Plot statistical metrics versus distance to coast.
- `plot_wave_height_*` – Various wave‑height visualisations (domain, building, error, histogram, scatter, etc.).
- `process_uplift_forces_elevated` – (Not detailed here) processing of uplift forces for elevated structures.
- `save_wave_stats` – Compute and save wave statistics to CSV/Dat files.
- `scratch` – Experimental or temporary scripts.
- `tmp_test_grid` – Test grid utilities.

For full details on each module, refer to the respective `README.md` inside the sub‑directory.

## Quick Start

1. Clone the repository and navigate to `xbeach-processing/`.
2. Edit `paths.txt` to point to your XBeach model run, forcing file, building GeoJSON, etc.
3. Open `main.py` and uncomment the block(s) corresponding to the figure or animation you wish to generate.
4. Run the script:
   ```sh
   python main.py
   ```
5. Results will be saved to the directory specified by `path_to_save_plot` in `paths.txt`.

---

*This README was generated automatically by aggregating the top‑level documentation and the individual module READMEs.*
