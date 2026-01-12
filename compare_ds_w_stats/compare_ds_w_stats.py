import os
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

from helpers.helpers import HelperFuncs
from save_wave_stats.save_wave_stats import SaveWaveStats


class CompareDSwStats(HelperFuncs):
    """docstring for xb_plotting_pt"""
    def __init__(self):
        super().__init__()
    
    def read_df(self, path_to_stats):
        df = pd.read_csv(path_to_stats)
        dmg = pd.read_csv(self.path_to_dmg)

        # dmg = dmg[["VDA_id", "VDA_DS_overall"]]
        dmg_unique = dmg.drop_duplicates(subset=["VDA_id"], keep="first")

        df.set_index("VDA_id", inplace=True)
        dmg_unique.set_index("VDA_id", inplace=True)

        df = pd.merge(df, dmg_unique, left_index=True, right_index=True)
        
        df = df.loc[~(df["surge_max"]>9)]
        return df

    def plot_violin(self, stats, path_to_stats, ncols=1, scatter=True, fname=None):
        df = self.read_df(path_to_stats)
        df["t_Hs_0.5m"] = df["t_Hs_0.5m"]/3600

        fig, ax = plt.subplots(int(len(stats)/ncols),ncols, figsize=(16,8))
        
        if isinstance(ax, np.ndarray):
            ax = ax.flatten()
        else:
            ax = [ax]
        colors = sns.color_palette("RdYlGn_r", 7)

        ds0 = df.loc[df["VDA_DS_overall"]=="DS0"]
        ds1 = df.loc[df["VDA_DS_overall"]=="DS1"]
        ds2 = df.loc[df["VDA_DS_overall"]=="DS2"]
        ds3 = df.loc[df["VDA_DS_overall"]=="DS3"]
        ds4 = df.loc[df["VDA_DS_overall"]=="DS4"]
        ds5 = df.loc[df["VDA_DS_overall"]=="DS5"]
        ds6 = df.loc[df["VDA_DS_overall"]=="DS6"]


        for stat_cnt, stat in enumerate(stats):
            ax_ = ax[stat_cnt]
            
            data_list = [ds0[stat].values, ds1[stat].values, ds2[stat].values, ds3[stat].values, 
                         ds4[stat].values, ds5[stat].values, ds6[stat].values]

            # Loop through the data, colors, and positions
            for i, data in enumerate(data_list):
                vplot = ax_.violinplot(data, positions=[i], showextrema=False)
                
                # Set the color for the current violin
                for body in vplot['bodies']:
                    body.set_facecolor(colors[i])
                    body.set_edgecolor('black')
                    body.set_linewidth(0.5)
                    body.set_alpha(0.8)

                x = self.rand_scatter(i, len(data))

                if scatter:
                    ax_.scatter(x, data, s=0.01, color='k', zorder=2)

            # x0 = self.rand_scatter(0,len(ds0))
            # x1 = self.rand_scatter(1,len(ds1))
            # x2 = self.rand_scatter(2,len(ds2))
            # x3 = self.rand_scatter(3,len(ds3))
            # x4 = self.rand_scatter(4,len(ds4))
            # x5 = self.rand_scatter(5,len(ds5))
            # x6 = self.rand_scatter(6,len(ds6))

            # if scatter:
            #     ax_.scatter(x0, ds0[stat].values, s=0.1, color='k', zorder=2)
            #     ax_.scatter(x1, ds1[stat].values, s=0.1, color='k', zorder=2)
            #     ax_.scatter(x2, ds2[stat].values, s=0.1, color='k', zorder=2)
            #     ax_.scatter(x3, ds3[stat].values, s=0.1, color='k', zorder=2)
            #     ax_.scatter(x4, ds4[stat].values, s=0.1, color='k', zorder=2)
            #     ax_.scatter(x5, ds5[stat].values, s=0.1, color='k', zorder=2)
            #     ax_.scatter(x6, ds6[stat].values, s=0.1, color='k', zorder=2)

            if stat == "impulse":
                lbl = "Impulse ((kN-hr)/m)"
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
    
    def temp_tree(self, df):
        from sklearn.datasets import load_iris
        from sklearn import tree

        x_cols = ["FFE_ffe_ft", "Hmax", "Hs_max", "impulse", "surge_max", "t_Hs_0.5m"]
        # x_cols = ["Hs_max", "t_Hs_0.5m"]
        y_col = ["VDA_DS_overall"]
        X = df[x_cols].values
        y = df[y_col].values
        y = np.ravel(y)
        print(len(y))
        clf = tree.DecisionTreeClassifier(max_depth=3, min_samples_split=3)
        clf = clf.fit(X, y)
        fig, ax = plt.subplots(1,1,figsize=(15,8))
        tree.plot_tree(clf, ax=ax, fontsize=5, feature_names=x_cols, class_names=np.unique(y),
                filled=True)
        
        plt.show()
        
    def explore_confusion(self, damaged_DSs=["DS5", "DS6"]):

        fn = os.path.join(self.path_to_save_plot, "removed_bldgs.csv")
        df_xbeach = pd.read_csv(fn)                         # read csv
        df_xbeach = df_xbeach.loc[df_xbeach["removed_bldgs"]!=-9999]    # remove buildings outside domain
        df_xbeach.set_index("VDA_id", inplace=True)         # set index

        df_dmg = pd.read_csv(self.path_to_dmg)              # read observations from VDA
        remove_bldgs = (df_dmg["FFE_elev_status"] == "elevated") & (df_dmg["FFE_foundation"]=="Piles/Columns")
        df_dmg = df_dmg.loc[~remove_bldgs]

        df_dmg.set_index("VDA_id", inplace=True)            # set index
        df_dmg["removed_vda"] = 0
        df_dmg.loc[df_dmg["VDA_DS_overall"].isin(damaged_DSs), "removed_vda"] = 1
        
        df_dmg['TA_ActYearBuilt_pre1970'] = False
        df_dmg.loc[df_dmg["TA_ActYearBuilt"]<1998, "TA_ActYearBuilt_pre1970"] = True
        # df_dmg.loc[df_dmg["TA_ActYearBuilt"]<1974, "TA_ActYearBuilt_pre1970"] = True

        # ---

        """ each column with observations for:
            [ALL]: all buildings vs. standing / not standing
            [MICRO]: false predictions in xbeach vs. observations.
        """
        # col = "NSI_bldgtype"          # [ALL] H (manufactured) destroyed   | [MICRO] -
        # col = "LC_occupancy_type"     # [ALL] manufactured homes destroyed | [MICRO] -
        # col = "VDA_breakaway_walls"   # [ALL] -                            | [MICRO] breakaway walls result in standing building ***
        # col = "TA_BldgUseTyp"         # [ALL] destroyed mobile homes       | [MICRO] -  
        col="TA_ActYearBuilt_pre1970" # [ALL] before 1970, more destroyed  | [MICRO] before 1970, more destroyed ***
        # col = "TA_EffYearBuilt"       # [ALL] -                            | [MICRO] before 1990, more destroyed
        # col = "FEC_Building_Use"      # [ALL] -                            | [MICRO] - 
        # col = "FFE_bldg_diagram"      # [ALL] "8" results in destroyed     | [MICRO] 1a (slab on grade) result in destroyed buildings

        # -- two plots
        df_dmg = df_dmg.dropna(subset=[col])
        x_vals = df_dmg[col].unique().tolist()
        df_observed_destroyd = df_dmg.loc[df_dmg["removed_vda"] == 1]
        df_observed_standing = df_dmg.loc[df_dmg["removed_vda"] == 0]
        
        fig1, ax = plt.subplots(1,2, figsize=(10,4))
        
        # df_observed_destroyd[col].hist(ax=ax[0], grid=False, xrot=45)
        # df_observed_standing[col].hist(ax=ax[1], grid=False, xrot=45)
        ax[0].set_title("destroyed")
        ax[1].set_title("standing")
        df_observed_destroyd[col].value_counts().reindex(x_vals,fill_value=0).plot.bar(ax=ax[0], grid=False, title="destroyed")
        df_observed_standing[col].value_counts().reindex(x_vals,fill_value=0).plot.bar(ax=ax[1], grid=False, title="standing")
        
        # self.save_fig(fig1, "all-{}" .format(col), dpi=1000)

        # -- four plots
        df = pd.merge(df_xbeach["removed_bldgs"], df_dmg["removed_vda"], left_index=True, right_index=True)

        df_true_standing = df.loc[(df["removed_bldgs"]==0) & (df["removed_vda"]==0)].index.to_list()
        df_true_destroyd = df.loc[(df["removed_bldgs"]==1) & (df["removed_vda"]==1)].index.to_list()
        df_false_standing = df.loc[(df["removed_bldgs"]==0) & (df["removed_vda"]==1)].index.to_list()
        df_false_destroyd = df.loc[(df["removed_bldgs"]==1) & (df["removed_vda"]==0)].index.to_list()
        
        df_true_standing = df_dmg.loc[df_true_standing]
        df_true_destroyd = df_dmg.loc[df_true_destroyd]
        df_false_standing = df_dmg.loc[df_false_standing]
        df_false_destroyd = df_dmg.loc[df_false_destroyd]
        
        fig2, ax = plt.subplots(2,2, figsize=(8,6))
        df_true_standing[col].value_counts().reindex(x_vals, fill_value=0).plot.bar(ax=ax[0,0], grid=False, title="True Standing")
        df_false_destroyd[col].value_counts().reindex(x_vals, fill_value=0).plot.bar(ax=ax[0,1], grid=False, title="False Destroyed")
        df_false_standing[col].value_counts().reindex(x_vals, fill_value=0).plot.bar(ax=ax[1,0], grid=False, title="False Standing")
        df_true_destroyd[col].value_counts().reindex(x_vals, fill_value=0).plot.bar(ax=ax[1,1], grid=False, title="True Destroyed")


        plt.tight_layout()
        # self.save_fig(fig2, "confusion-{}" .format(col), dpi=1000)
        plt.show()


    def plot_confusion(self, damaged_DSs=["DS5", "DS6"], count_elevated=False, fname=None):
        fn = os.path.join(self.path_to_save_plot, "removed_bldgs.csv")
        if os.path.exists(fn)==False:
            sws = SaveWaveStats()
            sws.save_removed_bldgs()
            sws.geolocate("removed_bldgs")
            sws.assign_to_bldgs(stats=["removed_bldgs"],
                            col_names=["removed_bldgs"],
                            runs=None,
                            fname="removed_bldgs.csv",
                            )
        df_xbeach = pd.read_csv(fn)                         # read csv
        df_xbeach = df_xbeach.loc[df_xbeach["removed_bldgs"]!=-9999]    # remove buildings outside domain
        df_xbeach.set_index("VDA_id", inplace=True)         # set index

        df_dmg = pd.read_csv(self.path_to_dmg)              # read observations from VDA
        elevated_bldgs = (df_dmg["FFE_elev_status"] == "elevated") & (df_dmg["FFE_foundation"]=="Piles/Columns")
        txt = "All buildings (including elevated)"
        if count_elevated==False:
            df_dmg = df_dmg.loc[~elevated_bldgs]
            txt = "Ignore Elevated"

        df_dmg.set_index("VDA_id", inplace=True)            # set index
        # set column for remove
        df_dmg["removed_vda"] = 0
        df_dmg.loc[df_dmg["VDA_DS_overall"].isin(damaged_DSs), "removed_vda"] = 1
        
        # merge two dataframes
        df = pd.merge(df_xbeach["removed_bldgs"], df_dmg["removed_vda"], left_index=True, right_index=True)

        # -- now create confusion matrix
        # Calculate the confusion matrix
        labels = [0,1]
        cm = confusion_matrix(df["removed_vda"], df["removed_bldgs"], labels=labels)
        score = (cm[0,0]+cm[1,1])/(np.sum(cm))
        score = "Percent Correct: {:0.3f}" .format(score)
        
        # -- create plot
        fig, ax = plt.subplots(figsize=(6, 4))
        im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Greys)
        ax.set_ylabel('Observed', fontsize=14, rotation=0)
        ax.set_xlabel('XBeach', fontsize=14)
        ax.text(x=1.0, y=1.01, s=score,transform=ax.transAxes, ha="right", va="bottom")
        ax.text(x=1.0, y=1.08, s=txt,  transform=ax.transAxes, ha="right", va="bottom")

        labels = ["Standing", "Destroyed"]
        tick_marks = range(len(labels))

        ax.set_xticks(tick_marks)
        ax.set_xticklabels(labels, rotation=0, fontsize=10)
        ax.set_yticks(tick_marks)
        ax.set_yticklabels(labels, rotation=0, fontsize=10)

        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(
                    j, i, cm[i, j],  # Position (j, i) and value
                    ha="center",     # Horizontal alignment
                    va="center",     # Vertical alignment
                    color="white" if cm[i, j] > thresh else "black", # Color logic
                    fontsize=12
                )
        ax.set_xticks(np.arange(-.5, len(labels), 1), minor=True)
        ax.set_yticks(np.arange(-.5, len(labels), 1), minor=True)

        
        ax.grid(which='minor', color='k', linestyle='-', linewidth=0.5)
        fig.tight_layout()
        self.save_fig(fig, fname, dpi=1000)














