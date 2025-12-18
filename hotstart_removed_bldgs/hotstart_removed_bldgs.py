import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy.ndimage as ndi

from helpers.helpers import HelperFuncs

class PlotRemovedBldgs(HelperFuncs):
    """docstring for plot_wave_heights"""
    def __init__(self):
        super().__init__()
        self.hotstart_runs = self.set_hotstart_runs()

    def plot(self, domain_size="estero", grey_background=False, cmap=None, fname=None):
        # H = self.read_npy(stat)
        removed_bldgs = self.read_removed_bldgs()
        removed_bldgs_mask = np.ma.array(removed_bldgs, mask=~removed_bldgs)

        model_dir = os.path.join(self.path_to_model, self.hotstart_runs[0])
        xgr, ygr, zgr = self.read_grid(model_dir)
        bldgs = self.read_buildings(model_dir)
        bldg_locs = ~np.ma.getmask(bldgs)

        # creating map to plot
        bldg_map = np.ma.array(bldgs, mask=np.ma.getmask(bldgs))
        bldg_map[bldg_locs] = 0
        bldg_map[removed_bldgs_mask] = 1

        cmap = mpl.colors.ListedColormap(["darkseagreen", "red"])

        if grey_background:
            cmap.set_bad("grey")
        else:
            cmap.set_bad(alpha=0.5)

        figsize = self.get_figsize(domain_size)
        fig, ax = plt.subplots(1,1, figsize=figsize)
        ax.pcolormesh(xgr, ygr, zgr, vmin=-8.5, vmax=8.5, cmap="BrBG_r", zorder=0)
        pcm = ax.pcolormesh(xgr, ygr, bldg_map, cmap=cmap, zorder=1)
        # ax.imshow(H_plot, cmap=cmap, origin="lower")

        # plt.colorbar(pcm, ax=ax, extend="max", label=labl, aspect=40)
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.set_aspect("equal")
        
        self.save_fig(fig, fname, transparent=True, dpi=1000)



