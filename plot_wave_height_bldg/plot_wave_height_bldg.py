import os
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy.ndimage as ndi


from helpers.helpers import HelperFuncs

class PlotWaveHeightBldg(HelperFuncs):
    """docstring for plot_wave_heights"""
    def __init__(self):
        super().__init__()

    def plot(self, stat, model_runname_w_bldgs=None, vmax=1, vmin=0, 
            domain_size="estero", grey_background=False, cmap=None, fname=None):
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

        labl = self.stat2labl(stat)

        figsize = self.get_figsize(domain_size)
        fig, ax = plt.subplots(1,1, figsize=figsize)
        ax.pcolormesh(xgr, ygr, zgr, vmin=-8.5, vmax=8.5, cmap="BrBG_r", zorder=0)
        pcm = ax.pcolormesh(xgr, ygr, bldg_H, vmin=vmin, vmax=vmax, cmap=cmap, zorder=1)


        plt.colorbar(pcm, ax=ax, extend="max", label=labl, aspect=40)
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.set_aspect("equal")
        
        self.save_fig(fig, fname, transparent=True, dpi=1000)

    def plot_geopandas(self, stat, 
            which_bldgs="elevated", 
            vmin=0, 
            vmax=1, 
            domain_size="estero",
            cmap="plasma", 
            fname=None):
        bldgs = self.read_bldgs_geodataframe()
        xo, yo, theta = self.get_origin()
        bldgs["geometry"] = bldgs["geometry"].rotate(angle=-theta, origin=(xo, yo))
        
        stats = os.path.join(self.path_to_save_plot, "stats_at_bldgs.csv")
        stats = pd.read_csv(stats)
        stats.set_index("VDA_id", inplace=True)

        bldgs = pd.merge(bldgs["geometry"], stats,  left_index=True, right_index=True)
        print(list(bldgs.columns))


        fn = os.path.join(self.path_to_save_plot, "forces_at_bldgs.csv")
        if (os.path.exists(fn)==False):
            sws = SaveWaveStats()
            sws.save_forces_at_bldg_to_csv()
        df_xbeach = pd.read_csv(fn)
        df_xbeach.set_index("VDA_id", inplace=True)
        bldgs = pd.merge(bldgs, df_xbeach["elevated"], left_index=True, right_index=True)
        print(bldgs)

        if which_bldgs=="non-elevated":
            bldgs = bldgs.loc[bldgs["elevated"]==False]
        elif which_bldgs=="elevated":
            bldgs = bldgs.loc[bldgs["elevated"]==True]
        else:
            raise ValueError("bldgs keyword must be: `all`, `elevated` or `non-elevated`")

        labl = self.stat2labl(stat)

        # -- plotting 
        figsize = self.get_figsize(domain_size)
        fig, ax = plt.subplots(1,1, figsize=figsize)

        bldgs.plot(ax=ax, 
            column=stat, 
            legend=False, 
            vmin=vmin, 
            vmax=vmax, 
            cmap=cmap,
            edgecolor='k',
            lw=0.05

            )
        mappable = ax.collections[0]
        plt.colorbar(mappable, ax=ax, extend="max", label=labl, aspect=40)

        self.remove_frame(ax)
        self.save_fig(fig, fname, dpi=2000)

    def stat2labl(self, stat):

        if stat == "Hs_max":
            labl = "Maximum Sig. Wave Height (m)"
        elif stat == "Hs":
            labl = "Sig. Wave Height (m)"
        elif stat == "Hs_tot":
            labl = "Total Sig. Wave Height (m)"
        elif stat == "zs_max":
            labl = "Maximum Water Elevation (m)"
        elif stat == "zs_mean":
            labl = "Mean Water Elevation (m)"    
        elif "t_Hs" in stat:
            labl = "Time Sig. Wave Height exceeds {} m (hr)" .format(stat.split("_")[-1].split("m")[0])
        elif stat == "Hmax":
            labl = "Max. Wave Height (m)"
        elif stat == "Tm":
            labl = "Mean Period (s)"
        elif "horizontal_impulse" in stat:
            labl = "Impulse ((KN-hr)/m)"
        return labl



