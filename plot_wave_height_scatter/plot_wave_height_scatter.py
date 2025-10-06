import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.ndimage as ndi


from helpers.helpers import HelperFuncs

class PlotWaveHeightScatter(HelperFuncs):
    """docstring for plot_wave_heights"""
    def __init__(self):
        super().__init__()

    def scatter_bldg(self, stat, runs, labels, plot_hist=True, run_w_bldgs=None, fname=None):
        xgr, ygr, zgr = self.read_grid()
        bldgs = self.read_buildings(run_w_bldgs)
        mask_bldgs = np.ma.getmask(bldgs)
        print("Need to resample array such that they are the same size when comparing 1 vs 2 m runs.")
        
        df = pd.DataFrame()
        runs.insert(0, self.model_runname)
        for run in runs:
            # read wave heights
            H = self.read_npy(stat, run)

            # assign wave height to buildlings
            bldg_H = self.assign_max_to_bldgs(H, bldgs)    # getting max at each building

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
        print("number bldgs with {}>{}: {}" .format(labels[0], labels[1], len(df_run0_greater)))
        print("number bldgs with {}<{}: {}" .format(labels[1], labels[0], len(df_run1_greater)))

        if plot_hist == True:
            fig, ax = plt.subplots(len(runs), len(runs), figsize=(9,8))        
            runs_short = [i.split("max")[0] for i in runs]
            ticks = [0.0, 0.5, 1.0, 1.5, 2.0]
            for col in range(len(runs)-1,-1,-1):
                for row in range(len(runs)-1,-1,-1):
                    if col>row:
                        self.remove_frame(ax[row, col])
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
                        ax[row,col].text(s=labels[col], x=0.98,y=0.9, transform=ax[row,col].transAxes, ha="right")
                        
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
                    ax[row,col].set_xlabel(labels[col], fontsize=8)
                    ax[row,col].set_ylabel(labels[row], fontsize=8)
        else:
            fig, ax = plt.subplots(1,1, figsize=(4,3.5))
            ticks = [0.0, 0.5, 1.0, 1.5, 2.0]
            
            x_data = df[runs[0]]
            y_data = df[runs[1]]
            
            ax.scatter(x_data, y_data, s=10, facecolor="None", edgecolor='k')
            ax.plot([-1,6], [-1,6], ls="-.", lw=1.0, zorder=1, color='k')
            ax.set_xlim([0,2])
            ax.set_ylim([0,2])

            ax.tick_params(axis='x', labelsize=7)
            ax.tick_params(axis='y', labelsize=7)

            ax.set_xticks(ticks)
            ax.set_yticks(ticks)

            # -- 
            ax.set_xlabel(labels[0], fontsize=10)
            ax.set_ylabel(labels[1], fontsize=10)
            ax.set_title("Significant Wave Height at Buildings (m)", fontsize=10)
        plt.subplots_adjust(wspace=0.2, hspace=0.2)
            # plt.xticks(rotation=45)
        self.save_fig(fig, fname, transparent=True, dpi=300)



    def scatter_domain(self, stat, runs, labels, plot_hist=True, fname=None):
        df = pd.DataFrame()
        runs.insert(0, self.model_runname)
        for run in runs:
            # read wave heights
            H = self.read_npy(stat, run)
            df[run] = H.flatten()

        if plot_hist == True:
            fig, ax = plt.subplots(len(runs), len(runs), figsize=(9,8))
            ticks = [0.0, 0.5, 1.0, 1.5, 2.0]
            for col in range(len(runs)-1,-1,-1):
                for row in range(len(runs)-1,-1,-1):
                    if col>row:
                        self.remove_frame(ax[row, col])
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
                        ax[row,col].text(s=labels[col], x=0.98,y=0.9, transform=ax[row,col].transAxes, ha="right")
                        
                        ax[row,col].get_yaxis().set_ticks([])
                        ax[row,col].tick_params(axis='x', labelsize=8)
                        continue

                    # -- if get to this point, then making scatter plot
                    x_data = df[runs[col]]
                    y_data = df[runs[row]]

                    ax[row,col].scatter(x_data, y_data, s=10, facecolor="None", edgecolor='k')
                    ax[row,col].plot([-1,6], [-1,6], ls="-.", lw=1.0, zorder=1, color='r')
                    ax[row,col].set_xlim([0,2])
                    ax[row,col].set_ylim([0,2])

                    ax[row,col].tick_params(axis='x', labelsize=7)
                    ax[row,col].tick_params(axis='y', labelsize=7)

                    ax[row,col].set_xticks(ticks)
                    ax[row,col].set_yticks(ticks)

                    # -- 
                    ax[row,col].set_xlabel(labels[col], fontsize=8)
                    ax[row,col].set_ylabel(labels[row], fontsize=8)
                    # --

                plt.subplots_adjust(wspace=0.2, hspace=0.2)
        else:
            fig, ax = plt.subplots(1,1, figsize=(4,3.5))
            ticks = [0.0, 0.5, 1.0, 1.5, 2.0]
            
            x_data = df[runs[0]]
            y_data = df[runs[1]]
            
            ax.scatter(x_data, y_data, s=10, facecolor="None", edgecolor='k', zorder=0)
            ax.plot([-1,6], [-1,6], ls="-.", lw=1.0, color='r', zorder=1)
            ax.set_xlim([0,2])
            ax.set_ylim([0,2])

            ax.tick_params(axis='x', labelsize=7)
            ax.tick_params(axis='y', labelsize=7)

            ax.set_xticks(ticks)
            ax.set_yticks(ticks)

            # -- 
            ax.set_xlabel(labels[0], fontsize=10)
            ax.set_ylabel(labels[1], fontsize=10)
            ax.set_title("Significant Wave Over Domain (m)", fontsize=10)    
        self.save_fig(fig, fname, transparent=True, dpi=300)




















