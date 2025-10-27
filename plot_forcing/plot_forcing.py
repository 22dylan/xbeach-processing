import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from helpers.helpers import HelperFuncs

class PlotForcing(HelperFuncs):
    def __init__(self, savepoint=4):
        super().__init__()
        self.loc_keys = {1: "sw", 2:"se", 3:"nw", 4:"ne", 5:"nearshore", 6:"offshore-central", 7:"onshore"}
        # self.fn_forcing = os.path.join(self.focring_dir, "xbeach{}-{}.dat" .format(savepoint, self.loc_keys[savepoint]))
        
    def plot(self, var, savepoint, duration=None, fname=None, figsize=(5,3)):
        label, ylabel, color = self.var2label(var)
        df = self.frcing_to_dataframe()

        start_idx = 0
        stop_idx = -1
        if duration!=None:
            t_start, t_stop = self.xbeach_duration_to_start_stop(duration)

        if t_start!= None:
            start_idx = df.loc[df["t_sec"]==t_start*3600].index[0]
        if t_stop!=None:
            stop_idx = df.loc[df["t_sec"]==t_stop*3600].index[0]
        
        df_trnc = df.iloc[start_idx:stop_idx]

        fig, ax = plt.subplots(1,1, figsize=figsize)
        # fig, ax = plt.subplots(1,1, figsize=(10,1.6))
        
        ls_full = "-"
        lw_full = 1.5
        if (t_start!=None) or (t_stop!=None):
            ax.plot(df_trnc["t_hr"], df_trnc[var], 
                    color="#ff5370", 
                    lw=3, 
                    label="XBeach", 
                    zorder=1)
            ls_full = "-."
            lw_full = 0.75

        ax.plot(df["t_hr"], df[var], color="k", lw=lw_full, ls=ls_full, label="ADCIRC/SWAN", zorder=0)
        ax.legend(loc="upper left")
        ax.set_xlabel("Time (Hours)")
        ax.set_ylabel(ylabel)
        ax.set_title("{}-sp{}-{}" .format(var, savepoint, self.loc_keys[savepoint]))
        ax.set_xlim([0,96])

        self.save_fig(fig, fname,
                        transparent=True,
                        dpi=500,
                        bbox_inches="tight",
                        pad_inches=0.1)


