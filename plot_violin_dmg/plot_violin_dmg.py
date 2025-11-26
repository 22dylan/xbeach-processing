import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from helpers.helpers import HelperFuncs

class PlotViolinDmg(HelperFuncs):
    """docstring for xb_plotting_pt"""
    def __init__(self):
        super().__init__()
    
    def plot(self, stats, path_to_stats, ncols=1, scatter=True, fname=None):
        df = pd.read_csv(path_to_stats)
        dmg = pd.read_csv(self.path_to_dmg)
        dmg = dmg[["TA_FolioID", "VDA_DS_overall"]]
        dmg_unique = dmg.drop_duplicates(subset=["TA_FolioID"], keep="first")

        df = pd.merge(df, dmg_unique, left_on="FolioID", right_on="TA_FolioID", how="inner")
        df = df.loc[~(df["impulse"]==0)]
        df = df.loc[~(df["surge_max"]>9)]


        fig, ax = plt.subplots(int(len(stats)/ncols),ncols, figsize=(16,8))
        
        if isinstance(ax, np.ndarray):
            ax = ax.flatten()
        else:
            ax = [ax]

        ds0 = df.loc[df["VDA_DS_overall"]=="DS0"]
        ds1 = df.loc[df["VDA_DS_overall"]=="DS1"]
        ds2 = df.loc[df["VDA_DS_overall"]=="DS2"]
        ds3 = df.loc[df["VDA_DS_overall"]=="DS3"]
        ds4 = df.loc[df["VDA_DS_overall"]=="DS4"]
        ds5 = df.loc[df["VDA_DS_overall"]=="DS5"]
        ds6 = df.loc[df["VDA_DS_overall"]=="DS6"]

        for stat_cnt, stat in enumerate(stats):
            ax_ = ax[stat_cnt]
            
            ax_.violinplot(ds0[stat].values, positions=[0], showextrema=False)
            ax_.violinplot(ds1[stat].values, positions=[1], showextrema=False)
            ax_.violinplot(ds2[stat].values, positions=[2], showextrema=False)
            ax_.violinplot(ds3[stat].values, positions=[3], showextrema=False)
            ax_.violinplot(ds4[stat].values, positions=[4], showextrema=False)
            ax_.violinplot(ds5[stat].values, positions=[5], showextrema=False)
            ax_.violinplot(ds6[stat].values, positions=[6], showextrema=False)

            x0 = self.rand_scatter(0,len(ds0))
            x1 = self.rand_scatter(1,len(ds1))
            x2 = self.rand_scatter(2,len(ds2))
            x3 = self.rand_scatter(3,len(ds3))
            x4 = self.rand_scatter(4,len(ds4))
            x5 = self.rand_scatter(5,len(ds5))
            x6 = self.rand_scatter(6,len(ds6))

            if scatter:
                ax_.scatter(x0, ds0[stat].values, s=0.01, color='k', zorder=2)
                ax_.scatter(x1, ds1[stat].values, s=0.01, color='k', zorder=2)
                ax_.scatter(x2, ds2[stat].values, s=0.01, color='k', zorder=2)
                ax_.scatter(x3, ds3[stat].values, s=0.01, color='k', zorder=2)
                ax_.scatter(x4, ds4[stat].values, s=0.01, color='k', zorder=2)
                ax_.scatter(x5, ds5[stat].values, s=0.01, color='k', zorder=2)
                ax_.scatter(x6, ds6[stat].values, s=0.01, color='k', zorder=2)

            if stat == "impulse":
                lbl = "Impulse (kN-hr)"
            elif stat == "Hs_max":
                lbl = "Hs_max (m)"
            elif stat == "Hs_tot":
                lbl = "Hs_tot (m)"
            elif stat == "Hmax":
                lbl = "H_max (m)"
            elif stat == "surge_max":
                lbl = "surge_max (m)"
            elif stat == "t_Hs_0.5m":
                lbl = "t_Hs > 0.5m (hr)"

            ax_.set_xlabel("Damage State")
            ax_.set_ylabel(lbl)
            ax_.set_xticks([0, 1, 2, 3, 4, 5, 6])
            ax_.set_xticklabels(["DS0", "DS1", "DS2", "DS3", "DS4", "DS5", "DS6"])
            ax_.grid(ls="-.", lw=0.5, zorder=0)
        plt.tight_layout()
        self.save_fig(fig, fname, dpi=1000)

    def rand_scatter(self, x, n):
        return np.ones(n)*x + 0.02*np.random.uniform(low=-1,high=1, size=n)

