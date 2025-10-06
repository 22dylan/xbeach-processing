import os
import numpy as np
import pandas as pd
import scipy.stats as st
import scipy.signal as sg
import xarray as xr
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from helpers.helpers import HelperFuncs

class PlotWaveHeights(HelperFuncs):
    """docstring for plot_wave_heights"""
    def __init__(self):
        super().__init__()

    def plot_wave_height_hist(self, runs, read_max_local=False, fname=None):        
        lbls = self.run2label_dict()
        df = pd.DataFrame()
        runname_list = []
        cnt = 0
        for run in runs:
            # read max wave heights
            run_max = self.read_npy(run)
            bldgs, mask, xgr, ygr, zgr = self.read_bldgs_grd(model_runname, read_max_local, run_max)
            mask_water = zgr>=0
            df[run] = run_max[mask_water]
            runname_list.append(model_runname)
            cnt += 1
        df = df.dropna(axis=0)  # removing all rows with nan

        # --- plotting histogram - comparing max wave heights, same plot
        fig, ax = plt.subplots(1,1, figsize=(8,4.5))
        colors = ["tan", "darkslategrey", "olive"]
        zorders = [1, 0, 2]
        alphas = [0.7, 0.7, 0.7]
        for run_i, run in enumerate(df.columns):
            print(run, zorders[run_i])
            rlbl = runname_list[run_i].split("-")[0]
            df[run].hist(ax=ax, 
                            bins=100, 
                            range=(0,2),
                            density=False, 
                            zorder=zorders[run_i], 
                            color=colors[run_i], 
                            edgecolor='black', 
                            linewidth=0.3,
                            label=lbls[rlbl],
                            alpha=alphas[run_i]
                            )

        ax.grid(False)
        ax.legend(bbox_to_anchor=(0.9, 0.95), frameon=False, facecolor=None)
        ax.set_xlabel("Max Wave Height (m)")
        ax.set_ylabel("Frequency")
        ax.set_xlim([0,2])
        # ---
        self.save_fig(fname, transparent=True, dpi=300)

































    def plot_wave_height_bldg_scatter(self, runs, model_runname_w_bldgs, 
                            readlocal=True, plt_same_axis=True, fname=None):

        # first getting mask where buildings are
        bldgs, xgr, ygr, zgr = self.read_bldgs_grd(model_runname_w_bldgs)
        # bldg_H = self.assign_max_to_bldgs(H, bldgs)
        mask_bldgs = np.ma.getmask(bldgs)
        print("Need to resample array such that they are the same size when comparing 1 vs 2 m runs.")
        
        df = pd.DataFrame()
        for run in runs:
            # read wave heights
            H = self.read_npy(run)

            run_ = run.split("_")[0]
            _, xgr, ygr, zgr = self.read_bldgs_grd(run_)
            
            # read buildings and grid 
            # bldgs, _, xgr, ygr, zgr = self.read_bldgs_grd(model_runname, readlocal, H)
            bldg_H = self.assign_max_to_bldgs(H, bldgs)    # getting max at each building
            # mask_bldgs = np.ma.getmask(bldgs)

            labeled_mask, num_features = ndi.label(~mask_bldgs)
            max_H_at_bldgs = []
            for i in range(num_features+1):
                if i == 0:
                    continue
                m_ = labeled_mask==i
                val = np.max(bldg_H[m_])
                max_H_at_bldgs.append(val.item())
            df[run] = max_H_at_bldgs

        df["diff"] = df[runs[0]] - df[runs[1]]
        df_run0_greater = df.loc[df["diff"]>0]
        df_run1_greater = df.loc[df["diff"]<0]
        print("number bldgs with run0>run1: {}" .format(len(df_run0_greater)))
        print("number bldgs with run0<run1: {}" .format(len(df_run1_greater)))

        if plt_same_axis:
            fig, ax = plt.subplots(1,1, figsize=(5,4))
            colors = ["k", "red"]
            for run_i, run in enumerate(runs[1:]):
                ax.scatter(df[runs[0]], df[run], facecolor="None", edgecolor="k", lw=1., s=20, zorder=0)

                # drawing 1 to 1 line
                ax.plot([-1,6], [-1,6], ls="-.", lw=1.0, zorder=1, color='k', label="1-to-1")

            #------------------
            lbls = self.run2label_dict()
            
            ax.set_xlabel("{}" .format(lbls[runs[0].split("max")[0]]))
            ax.set_ylabel("{}" .format(lbls[runs[1].split("max")[0]]))

            ax.set_xlim([0,3])
            ax.set_ylim([0,3])
            # ax.legend(loc="upper left")
            ax.set_title("Significant Wave Height at Buildings (m)")
        
            # regression to r^2 and best fit line
            slope, intercept, r_value, p_value, std_err = st.linregress(df[runs[0]], df[run])
            x = np.linspace(0,6, 100)
            y = slope*x + intercept
            # ax.plot(x,y, ls="-", lw=1.5, color="purple", label="Regression")

            s1 = "Slope = {:0.4f}\n" .format(slope)
            s2 = "Intercept = {:0.4f}\n" .format(intercept)
            s3 = r"$r^2= $ {:0.4f}" .format(r_value)
            s = s1+s2+s3

            ax.text(x=0.05, y=0.95, s=s, 
                    transform=ax.transAxes, 
                    horizontalalignment='left', 
                    verticalalignment="top",
                    bbox=dict(facecolor='none', edgecolor='k'))

        else:
            fig, ax = plt.subplots(len(runs), len(runs), figsize=(9,8))
            # self.remove_frame(ax[0,0])
            self.remove_frame(ax[0,1])
            self.remove_frame(ax[0,2])
            # self.remove_frame(ax[1,1])
            self.remove_frame(ax[1,2])
            # self.remove_frame(ax[2,2])
            lbls = self.run2label_dict()
            runs_short = [i.split("max")[0] for i in runs]

            ticks = [0.0, 0.5, 1.0, 1.5, 2.0]
            for col in range(len(runs)-1,-1,-1):
                for row in range(len(runs)-1,-1,-1):
                    if col>row:
                        continue
                    if col==row:
                        df[runs[col]].hist(
                            ax=ax[row, col], 
                            bins=20,
                            range=(0,2),
                            density=True,
                            color="tan",
                            edgecolor='black',
                            linewidth=0.3,
                            alpha=0.7
                            )
                        df[runs[col]].plot.kde(
                            ax=ax[row,col],
                            color='k',
                            lw=1
                            )
                        ax[row,col].set_xlabel(None)
                        ax[row,col].set_ylabel(None)
                        ax[row,col].grid(False)
                        ax[row,col].set_xlim([0,2])
                        ax[row,col].text(s=lbls[runs_short[col]], x=0.98,y=0.9, transform=ax[row,col].transAxes, ha="right")
                        
                        ax[row,col].get_yaxis().set_ticks([])
                        ax[row,col].tick_params(axis='x', labelsize=8)
                        continue

                    # -- if get to this point, then making scatter plot
                    x_data = df[runs[col]]
                    y_data = df[runs[row]]

                    ax[row,col].scatter(x_data, y_data, s=10, facecolor="None", edgecolor='k')
                    ax[row,col].plot([-1,6], [-1,6], ls="-.", lw=1.0, zorder=1, color='k')
                    ax[row,col].set_xlim([0,2])
                    ax[row,col].set_ylim([0,2])

                    ax[row,col].tick_params(axis='x', labelsize=7)
                    ax[row,col].tick_params(axis='y', labelsize=7)

                    ax[row,col].set_xticks(ticks)
                    ax[row,col].set_yticks(ticks)

                    # -- 
                    ax[row,col].set_xlabel(lbls[runs_short[col]], fontsize=8)
                    ax[row,col].set_ylabel(lbls[runs_short[row]], fontsize=8)
                    # --


            plt.subplots_adjust(wspace=0.2, hspace=0.2)
            # plt.xticks(rotation=45)
        self.save_fig(fname, transparent=True, dpi=300)
    
    def print_summary_statistics(self, runs, model_runname_w_bldgs=None, readlocal=True):
        if model_runname_w_bldgs != None:
            # first getting mask where buildings are
            H_ = self.read_npy(model_runname_w_bldgs)
            _, mask_bldgs, xgr, ygr, zgr = self.read_bldgs_grd(model_runname_w_bldgs, readlocal, H_)

        df = pd.DataFrame()
        for run in runs:
            # read wave heights
            H = self.read_npy(run)
            
            if model_runname_w_bldgs != None:
                # read buildings and grid 
                bldg_H = self.assign_max_to_bldgs(H, mask_bldgs)    # getting max at each building
            
                labeled_mask, num_features = ndi.label(~mask_bldgs)
                max_H_at_bldgs = []
                for i in range(num_features+1):
                    if i == 0:
                        continue
                    m_ = labeled_mask==i
                    val = np.max(bldg_H[m_])
                    max_H_at_bldgs.append(val.item())

                df[run] = max_H_at_bldgs
            else:
                _, _, xgr, ygr, zgr = self.read_bldgs_grd(model_runname, readlocal, H)
                # --
                mask_water = zgr>=0
                df[run] = H[mask_water]
                # --        

        df = df.dropna(axis=0)  # removing all rows with nan

        lbl = self.run2label_dict()
        runs_short = [i.split("max")[0] for i in runs]
        
        print(df.head())
        

        print("\n-------------------")
        print("run15 avg: {:0.3f}" .format(df["run15max.npy"].mean()))
        print("run16 avg: {:0.3f}" .format(df["run16max.npy"].mean()))
        print("run18 avg: {:0.3f}" .format(df["run18max.npy"].mean()))
        print("")

        diff_r18r15 = df["run18max.npy"] - df["run15max.npy"]
        print("\n-------------------")
        print("{} ({}) vs. {} ({})" .format(runs_short[0], lbl[runs_short[0]], runs_short[2], lbl[runs_short[2]]))
        print("Avg.: {:0.3f}" .format(diff_r18r15.mean().item()))
        print("Med.: {:0.3f}" .format(diff_r18r15.median().item()))
        print("Max:  {:0.3f}" .format(diff_r18r15.max().item()))
        print("Min:  {:0.3f}" .format(diff_r18r15.min().item()))
        if model_runname_w_bldgs!=None:
            t = diff_r18r15>0
            print("Num bldgs with run18>run15: {}" .format(t.sum()))
            print("Num bldgs with run18<run15: {}" .format((~t).sum()))
        print("")

        diff_r18r16 = df["run18max.npy"] - df["run16max.npy"]
        print("{} ({}) vs. {} ({})" .format(runs_short[1], lbl[runs_short[1]], runs_short[2], lbl[runs_short[2]]))
        print("Avg.: {:0.3f}" .format(diff_r18r16.mean().item()))
        print("Med.: {:0.3f}" .format(diff_r18r16.median().item()))
        print("Max:  {:0.3f}" .format(diff_r18r16.max().item()))
        print("Min:  {:0.3f}" .format(diff_r18r16.min().item()))
        if model_runname_w_bldgs!=None:
            t = diff_r18r16>0
            print("Num bldgs with run18>run16: {}" .format(t.sum()))
            print("Num bldgs with run18<run16: {}" .format((~t).sum()))
        print("-------------------\n")


    def plot_wave_height_scatter(self, runs, readlocal=True, plt_same_axis=True, fname=None):
        df = pd.DataFrame()
        for run in runs:
            # read wave heights
            H = self.read_npy(run)
            df[run] = H

        if plt_same_axis:
            fig, ax = plt.subplots(1,1, figsize=(5,4))
            for run_i, run in enumerate(runs[1:]):
                ax.scatter(df[runs[0]], df[run], facecolor="None", edgecolor="k", lw=1., s=20, zorder=0)
                ax.plot([-1,6], [-1,6], ls="-.", lw=1.0, zorder=1, color='k', label="1-to-1")

            #------------------
            lbls = self.run2label_dict()
            
            ax.set_xlabel("{}" .format(lbls[runs[0].split("max")[0]]))
            ax.set_ylabel("{}" .format(lbls[runs[1].split("max")[0]]))

            ax.set_xlim([0,3])
            ax.set_ylim([0,3])
            # ax.legend(loc="upper left")
            ax.set_title("Significant Wave Height at Buildings (m)")
        
            # regression to r^2 and best fit line
            slope, intercept, r_value, p_value, std_err = st.linregress(df[runs[0]], df[run])
            x = np.linspace(0,6, 100)
            y = slope*x + intercept
            # ax.plot(x,y, ls="-", lw=1.5, color="purple", label="Regression")

            s1 = "Slope = {:0.4f}\n" .format(slope)
            s2 = "Intercept = {:0.4f}\n" .format(intercept)
            s3 = r"$r^2= $ {:0.4f}" .format(r_value)
            s = s1+s2+s3

            ax.text(x=0.05, y=0.95, s=s, 
                    transform=ax.transAxes, 
                    horizontalalignment='left', 
                    verticalalignment="top",
                    bbox=dict(facecolor='none', edgecolor='k'))

        else:
            fig, ax = plt.subplots(len(runs), len(runs), figsize=(9,8))
            # self.remove_frame(ax[0,0])
            self.remove_frame(ax[0,1])
            self.remove_frame(ax[0,2])
            # self.remove_frame(ax[1,1])
            self.remove_frame(ax[1,2])
            # self.remove_frame(ax[2,2])
            lbls = self.run2label_dict()
            runs_short = [i.split("max")[0] for i in runs]

            ticks = [0.0, 0.5, 1.0, 1.5, 2.0]
            for col in range(len(runs)-1,-1,-1):
                for row in range(len(runs)-1,-1,-1):
                    if col>row:
                        continue
                    if col==row:
                        df[runs[col]].hist(
                            ax=ax[row, col], 
                            bins=20,
                            range=(0,2),
                            density=True,
                            color="tan",
                            edgecolor='black',
                            linewidth=0.3,
                            alpha=0.7
                            )
                        df[runs[col]].plot.kde(
                            ax=ax[row,col],
                            color='k',
                            lw=1
                            )
                        ax[row,col].set_xlabel(None)
                        ax[row,col].set_ylabel(None)
                        ax[row,col].grid(False)
                        ax[row,col].set_xlim([0,2])
                        ax[row,col].text(s=lbls[runs_short[col]], x=0.98,y=0.9, transform=ax[row,col].transAxes, ha="right")
                        
                        ax[row,col].get_yaxis().set_ticks([])
                        ax[row,col].tick_params(axis='x', labelsize=8)
                        continue

                    # -- if get to this point, then making scatter plot
                    x_data = df[runs[col]]
                    y_data = df[runs[row]]

                    ax[row,col].scatter(x_data, y_data, s=10, facecolor="None", edgecolor='k')
                    ax[row,col].plot([-1,6], [-1,6], ls="-.", lw=1.0, zorder=1, color='k')
                    ax[row,col].set_xlim([0,2])
                    ax[row,col].set_ylim([0,2])

                    ax[row,col].tick_params(axis='x', labelsize=7)
                    ax[row,col].tick_params(axis='y', labelsize=7)

                    ax[row,col].set_xticks(ticks)
                    ax[row,col].set_yticks(ticks)

                    # -- 
                    ax[row,col].set_xlabel(lbls[runs_short[col]], fontsize=8)
                    ax[row,col].set_ylabel(lbls[runs_short[row]], fontsize=8)
                    # --


            plt.subplots_adjust(wspace=0.2, hspace=0.2)
            # plt.xticks(rotation=45)
        self.save_fig(fname, transparent=True, dpi=300)


    def run2label_dict(self):
        d = {
            "run15": "buildings-on-ground",
            "run16": "remove-elevated",
            "run17": "remove-first-row",
            "run18": "no-buildings",
            # "run19": "",
            "run20": "5 s output sampling",
            "run21": "2 s output sampling",
            "temp": "test",
            "run26_Hs.npy": "buildings-on-ground",
            "run27-copy_Hs.npy": "1 m resolution",
            "run31_Hs.npy": "no-buildings",
        }
        return d





# if __name__ == "__main__":
    
#     pwh = plot_wave_heights(var="zs1")
#     # pwh.save_wave_stats("run26", stat="Hs")

#     pwh.plot_wave_height_domain(model_runname="run27-copy_Hs.npy", 
#                              readlocal=True,
#                              vmin=0,
#                              vmax=1,
#                              single_frame=True, 
#                              domain_size="micro",
#                              plt_bldgs=True,
#                              fname="run27-copy_Hs.png"
#                              )

    # pwh.plot_wave_height_bldg(
    #                           model_runname="run26_Hs.npy",
    #                           model_runname_w_bldgs="run26",
    #                           readlocal=True,                              
    #                           vmax=1,
    #                           domain_size="micro",  # or estero
    #                           grey_background=True,
    #                           fname="run26-Hs-bldgs.png"
    #                         )

    # pwh.plot_wave_height_bldg_scatter(
    #                               runs = ["run26_Hs.npy", "run31_Hs.npy"],
    #                               model_runname_w_bldgs="run26",
    #                               readlocal=True,
    #                               plt_same_axis=True,
    #                               fname="temp.png"
    #                             )

    # pwh.print_summary_statistics(
    #                             runs=["run15max.npy", "run16max.npy", "run18max.npy"],
    #                             model_runname_w_bldgs="run16max.npy",
    #                             readlocal=True
    #     )

    # pwh.plot_wave_height_domain_diff(run1="run15max.npy", 
    #                               run2="run16max.npy", 
    #                               r1local=True, 
    #                               r2local=True, 
    #                               vmax=1, 
    #                               norm=False,
    #                               domain_size="micro",
    #                               fname="diff_r15r16_map.png"
    #                               )

    # pwh.plot_wave_height_scatter(runs=["run"],
    #                             readlocal=True,
    #                             plt_same_axis=True
    #                              # fname="temp.png"
    #                              )

    # pwh.plot_wave_height_hist(
    #                        runs = ["run15max.npy", "run16max.npy", "run18max.npy"],
    #                        read_max_local=True, 
    #                        fname="r15r16r18_hist.png"
    #                        )



    plt.show()

