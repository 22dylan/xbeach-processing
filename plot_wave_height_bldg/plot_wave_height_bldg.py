import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy.ndimage as ndi


from helpers.helpers import HelperFuncs

class PlotWaveHeightBldg(HelperFuncs):
    """docstring for plot_wave_heights"""
    def __init__(self):
        super().__init__()

    def plot(self, stat, model_runname_w_bldgs=None, vmax=1, vmin=0, 
            domain_size="estero", grey_background=False, fname=None):
        # read wave heights
        H = self.read_npy(stat)
        
        xgr, ygr, zgr = self.read_grid()
        bldgs = self.read_buildings()

        # assign max H to each building
        bldg_H = self.assign_max_to_bldgs(H, bldgs)

        # setting up cmap
        cmap = mpl.cm.plasma
        if grey_background:
            cmap.set_bad("grey")
        else:
            cmap.set_bad(alpha=0.5)

        figsize = self.get_figsize(domain_size)        
        fig, ax = plt.subplots(1,1, figsize=figsize)
        ax.pcolormesh(xgr, ygr, zgr, vmin=-8.5, vmax=8.5, cmap="BrBG_r", zorder=0)
        pcm = ax.pcolormesh(xgr, ygr, bldg_H, vmin=vmin, vmax=vmax, cmap=cmap, zorder=1)
        plt.colorbar(pcm, ax=ax, extend="max", label="Max Wave Height (m)", aspect=40)
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")

        self.save_fig(fig, fname, transparent=True, dpi=1000)


