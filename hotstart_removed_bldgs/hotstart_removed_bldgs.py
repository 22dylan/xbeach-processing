import os
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib as mpl
import matplotlib.pyplot as plt

from helpers.helpers import HelperFuncs
from save_wave_stats.save_wave_stats import SaveWaveStats

class PlotRemovedBldgs(HelperFuncs):
    """docstring for plot_wave_heights"""
    def __init__(self):
        super().__init__()
        self.hotstart_runs = self.set_hotstart_runs()

    def plot_geopandas(self, 
            bldgs="all", 
            domain_size="estero", 
            elevated_kwds=None,
            fname=None):
        fn = os.path.join(self.path_to_save_plot, "removed_bldgs.csv")
        if (os.path.exists(fn)==False) or (elevated_kwds["compute_removed_elevated"]==True):
            sws = SaveWaveStats()
            sws.save_removed_bldgs()
            sws.save_removed_elevated_bldgs(threshold=elevated_kwds["removed_elevated_threshold"])
            sws.geolocate("removed_bldgs")
            sws.geolocate("removed_bldgs_elevated")
            sws.assign_to_bldgs(stats=["removed_bldgs", "removed_bldgs_elevated"],
                            col_names=["removed_bldgs_non_elevated", "removed_bldgs_elevated"],
                            runs=None,
                            fname="removed_bldgs.csv",
                            )
            sws.merge_remove_bldgs()

        df_xbeach = pd.read_csv(fn)
        df_xbeach.set_index("VDA_id", inplace=True)

        gdf_bldgs = gpd.read_file(self.path_to_bldgs)
        gdf_bldgs.set_index("VDA_id", inplace=True)

        gdf_bldgs = pd.merge(gdf_bldgs, df_xbeach, left_index=True, right_index=True)

        # -- rotate geodataframe
        xo, yo, theta = self.get_origin()
        gdf_bldgs["geometry"] = gdf_bldgs["geometry"].rotate(angle=-theta, origin=(xo, yo))

        # -- new
        if bldgs=="all":
            txt = "All buildings (including elevated)"
            gdf_elevated = gdf_bldgs.loc[gdf_bldgs["elevated"]==True]
            edgecolor = "none"
        elif bldgs=="non-elevated":
            gdf_bldgs = gdf_bldgs.loc[gdf_bldgs["elevated"]==False]
            txt = "Ignore Elevated"
            edgecolor = "none"
        elif bldgs=="elevated":
            gdf_bldgs = gdf_bldgs.loc[gdf_bldgs["elevated"]==True]
            txt = "Elevated Only"
            edgecolor = "k"
        else:
            raise ValueError("bldgs keyword must be: `all`, `elevated` or `non-elevated`")

        # ---


        figsize = self.get_figsize(domain_size)
        fig, ax = plt.subplots(1,1, figsize=figsize)
        cmap = mpl.colors.ListedColormap(["darkseagreen", "red"])
        
        gdf_bldgs.plot(ax=ax, column="removed_bldgs", cmap=cmap, edgecolor=edgecolor)
        ax.get_xaxis().set_ticks([])
        ax.get_yaxis().set_ticks([])

        if (bldgs == "all"):
            gdf_elevated.plot(
                    ax=ax,
                    color="none",
                    edgecolor='k',
                    legend_kwds={"labels":["Elevated"]}
                    )
        
        self.save_fig(fig, fname, transparent=True, dpi=1000)

    def plot(self, domain_size="estero", include_elevated=False, grey_background=False, cmap=None, fname=None):
        # H = self.read_npy(stat)
        removed_bldgs = self.read_removed_bldgs()
        removed_bldgs_mask = np.ma.array(removed_bldgs, mask=~removed_bldgs)

        xgr, ygr, zgr = self.read_grid()
        bldgs = self.read_buildings()
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
        
        # plt.colorbar(pcm, ax=ax, extend="max", label=labl, aspect=40)
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.set_aspect("equal")
        
        self.save_fig(fig, fname, transparent=True, dpi=1000)



