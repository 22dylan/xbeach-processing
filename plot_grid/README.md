# Plot Grid

The **`PlotGrid`** class provides utilities for visualising the XBeach model grid.

## New `plot_grid` method

```python
pg = PlotGrid()
pg.plot_grid(
    cmap="viridis",   # colormap (default)
    vmin=None,         # optional lower colour limit
    vmax=None,         # optional upper colour limit
    savefig=False,     # set ``True`` to write the figure to disk
    fname="grid.png", # filename used when ``savefig=True``
)
```

* Reads the `x.grd`, `y.grd`, and `z.grd` files via the helper functions.
* Produces a full‑grid colour‑mesh plot.
* **Aspect ratio is now forced to be equal** (`ax.set_aspect('equal')`) so the grid is displayed without distortion.
* If `savefig=True`, the figure is saved using the common `HelperFuncs.save_fig` routine.

## Existing functionality

`plot_dep_across_y` remains unchanged – it plots elevation profiles across the y‑direction for a list of x‑transverse locations.

---

Make sure the paths in `paths.txt` point to a directory containing the required `x.grd`, `y.grd`, and `z.grd` files before calling the methods.
