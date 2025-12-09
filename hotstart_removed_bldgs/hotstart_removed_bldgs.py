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

    def plot(self, stat, threshold=None, vmax=1, vmin=0, 
            domain_size="estero", grey_background=False, cmap=None, fname=None):
        # read wave heights
        H = self.read_npy(stat)
        
        model_dir = os.path.join(self.path_to_model, "test_a1")
        xgr, ygr, zgr = self.read_grid(model_dir)
        bldgs = self.read_buildings(model_dir)
        
        bldg_locs = ~np.ma.getmask(bldgs)
        
        H_plot = np.ones(np.shape(H))*9999
        H_plot[(H>=threshold)&(bldg_locs)] = 1        
        H_plot[(H<threshold)&(bldg_locs)] = 0
        H_plot[~bldg_locs] = np.nan

        # setting up cmap
        # cmap = mpl.cm.plasma
        cmap = mpl.colors.ListedColormap(["darkseagreen", "red"])

        if grey_background:
            cmap.set_bad("grey")
        else:
            cmap.set_bad(alpha=0.5)

        figsize = self.get_figsize(domain_size)
        fig, ax = plt.subplots(1,1, figsize=figsize)
        ax.pcolormesh(xgr, ygr, zgr, vmin=-8.5, vmax=8.5, cmap="BrBG_r", zorder=0)
        pcm = ax.pcolormesh(xgr, ygr, H_plot, cmap=cmap, zorder=1)
        # ax.imshow(H_plot, cmap=cmap, origin="lower")

        # plt.colorbar(pcm, ax=ax, extend="max", label=labl, aspect=40)
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.set_aspect("equal")
        
        self.save_fig(fig, fname, transparent=True, dpi=1000)




