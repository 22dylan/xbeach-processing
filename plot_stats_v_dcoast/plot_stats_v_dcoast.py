import os
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib as mpl
import matplotlib.pyplot as plt

from helpers.helpers import HelperFuncs

class PlotStatsVDCoast(HelperFuncs):
    """docstring for plot_wave_heights"""
    def __init__(self):
        super().__init__()

    def plot(self, stats_plot, which_bldgs="all", remove_DSs=["DS6"], fname=None):
        
        bldgs = self.read_bldgs_geodataframe()
        
        coast = self.read_coast()
        coast.to_crs(bldgs.crs, inplace=True)
        coast = coast.iloc[0].geometry

        stats = os.path.join(self.path_to_save_plot, "stats_at_bldgs.csv")
        stats = pd.read_csv(stats)
        stats.set_index("VDA_id", inplace=True)

        bldgs = pd.merge(bldgs["geometry"], stats,  left_index=True, right_index=True)
        bldgs['dcoast'] = bldgs.geometry.distance(coast)

        # -- read removed bldg stats:
        stats2 = os.path.join(self.path_to_save_plot, "forces_at_bldgs.csv")
        stats2 = pd.read_csv(stats2)
        stats2.set_index("VDA_id", inplace=True)

        bldgs = pd.merge(bldgs, stats2["elevated"], left_index=True, right_index=True)

        dmg   =  pd.read_csv(self.path_to_dmg)
        dmg.set_index("VDA_id", inplace=True)
        dmg["removed"] = 0
        dmg["TA_ShapeSTArea_Sqm"] = dmg["TA_ShapeSTArea_Sqft"]*(0.3048*0.3048)

        dmg.loc[dmg["VDA_DS_overall"].isin(remove_DSs), "removed"] = 1
        bldgs = pd.merge(bldgs, dmg[["removed", "FFE_ffe_ft", "TA_ShapeSTArea_Sqm"]], left_index=True, right_index=True)
        bldgs["color"] = ['red' if r==1 else 'darkseagreen' for r in bldgs["removed"]]
        
        if which_bldgs == "all":
            bldgs_plot = bldgs
        elif which_bldgs == "elevated":
            bldgs_plot = bldgs.loc[bldgs["elevated"]==True]
        elif which_bldgs == "non-elevated":
            bldgs_plot = bldgs.loc[bldgs["elevated"]==False]


        # fig, ax = plt.subplots(2,int(len(stats_plot)/2), figsize=(10,5))
        fig, ax = plt.subplots(3,4, figsize=(12,8))
        ax = ax.flatten()
        for stat_i, stat in enumerate(stats_plot):
            if "uplift" in stat:
                bldgs_plot[stat] = bldgs_plot[stat]/bldgs_plot["TA_ShapeSTArea_Sqm"]
            ax[stat_i].scatter(bldgs_plot["dcoast"], bldgs_plot[stat], s=1, color=bldgs_plot["color"])
            ax[stat_i].set_xlabel("Distance to coast (m)", fontsize=8)
            labl = self.stat_to_label(stat)
            ax[stat_i].set_ylabel(labl, fontsize=8)
            ax[stat_i].tick_params(axis='both', which='major', labelsize=7)
            ax[stat_i].tick_params(axis='both', which='minor', labelsize=7)

        fb_surge = bldgs_plot["stat_water_elev_out"] - (bldgs_plot["FFE_ffe_ft"]*0.3048)
        fb_surge_wave = bldgs_plot["stat_max_zs"] - (bldgs_plot["FFE_ffe_ft"]*0.3048)

        fb_surge[fb_surge<0] = 0
        fb_surge_wave[fb_surge_wave<0] = 0

        ax[stat_i+1].scatter(bldgs_plot["dcoast"], fb_surge_wave, s=1, color=bldgs_plot["color"])
        ax[stat_i+1].set_ylabel("Freeboard (surge and wave) (m)", fontsize=8)
        ax[stat_i+1].tick_params(axis='both', which='major', labelsize=7)
        ax[stat_i+1].tick_params(axis='both', which='minor', labelsize=7)

        ax[stat_i+2].scatter(bldgs_plot["dcoast"], fb_surge, s=1, color=bldgs_plot["color"])
        ax[stat_i+2].set_ylabel("Freeboard (surge only) (m)", fontsize=8)
        ax[stat_i+2].tick_params(axis='both', which='major', labelsize=7)
        ax[stat_i+2].tick_params(axis='both', which='minor', labelsize=7)

        self.remove_frame(ax[stat_i+3])
        self.remove_frame(ax[stat_i+4])
        plt.tight_layout()
        self.save_fig(fig, fname, transparent=False, dpi=1000)




    def stat_to_label(self, stat):
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
        # elif "t_Hs" in stat:
            # labl = "Time Sig. Wave Height exceeds {} m (hr)" .format(stat.split("_")[-1].split("m")[0])
        elif stat == "Hmax":
            labl = "Max. Wave Height (m)"
        elif stat == "Tm":
            labl = "Mean Period (s)"
        elif stat == "impulse":
            labl = "Impulse ((KN-hr)/m)"
        
        elif stat == "stat_water_elev_out":
            labl = "Max Surge (m)"
        elif stat == "stat_uplift_impulse":
            labl = "Max Uplift Impulse\n15 Minute Chunk (kN-hr)"
        elif stat == "stat_max_zs":
            labl = "Max Water Level (surge + wave) (m)"
        elif stat == "stat_horizontal_impulse":
            labl = "Max Wave Impulse\n15 Minute Chunk (kN-hr/m)"
        elif stat == "stat_Hs":
            labl = "Max Significant Wave Height (m)"
        elif stat == "stat_cumulative_horizontal_impulse":
            labl = "Cumulative Wave\nImpulse (kN-hr/m)"
        elif stat == "stat_Hmax":
            labl = "Maximum Wave Height (m)"
        elif stat == "stat_cumulative_uplift_impulse":
            labl = "Cumulative Uplift Impulse (kN-hr)"

        else:
            labl = "Label not created yet"

        return labl














