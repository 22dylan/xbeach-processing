import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.ndimage as ndi


from helpers.helpers import HelperFuncs

class PlotWaveHeightScatter(HelperFuncs):
    """docstring for plot_wave_heights"""
    def __init__(self):
        super().__init__()

    def scatter_bldg(self, stat, runs, labels, plot_hist=True, run_w_bldgs=None, 
                    crude=False, lim=3, fname=None):
        """ only use this when trying to compare runs with different size domains. 
            This code is so poorly written. not intended to be generalized, so be careful.
        """
        if crude:
            # -- getting buildings / location of buildings
            bldgs = self.read_buildings(run_w_bldgs)

            print("this is hideously done")
            first_row = bldgs[0:1, :]
            last_row =  bldgs[-1:, :]
            bldgs = np.vstack((first_row, bldgs, last_row))
            bldgs = bldgs[:,:-2]
            mask = (bldgs != 10)
            bldgs = np.ma.array(bldgs, mask=mask)

            mask_bldgs = np.ma.getmask(bldgs)

            # -- get size of each domain that is being compared
            runs.insert(0, self.model_runname)      # list of runs
            sizes = []                  # list of domain sizes
            wave_heights = {}           # dict to store wave heights
            for run in runs:            # loop through runs
                H = self.read_npy(stat, run)    # read wave height (2D)
                if run == "s1":
                    print("This is very crudely done; trimming s1 to exact dimensions of other runs")
                    first_row = H[0:1, :]
                    last_row =  H[-1:, :]
                    H = np.vstack((first_row, H, last_row))
                    H = H[:,:-2]
                elif run == "s2":
                    print("This is very crudely done; trimming s2 to exact dimensions of other runs")
                    first_row = H[0:1, :]
                    H = np.vstack((first_row, H))
                    H = H[:,:-1]

                sizes.append(np.shape(H)[0])
                wave_heights[run] = H

            largest_size_idx = np.argmax(sizes)
            H_domain = wave_heights[runs[largest_size_idx]] # domain size that we'll use to compare runs

            df = pd.DataFrame()
            for run in runs:
                _, H = self.check_domain_size_wave_stat(H_domain, wave_heights[run])
                
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

        # ---------------------------------------------------
        else:
            bldgs = self.read_buildings(run_w_bldgs)
            mask_bldgs = np.ma.getmask(bldgs)
            print("Need to resample array such that they are the same size when comparing 1 vs 2 m runs.")
            
            df = pd.DataFrame()
            runs.insert(0, self.model_runname)
            
            sizes = []
            for run in runs:
                H = self.read_npy(stat, run)
                sizes.append(np.shape(H)[0])
            
            largest_size_idx = np.argmax(sizes)
            H_domain = self.read_npy(stat, runs[largest_size_idx])

            for run in runs:
                # read wave heights
                H = self.read_npy(stat, run)
                _, H = self.check_domain_size_wave_stat(H_domain, H)

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
            # ----------------------------------------------------------------

        if plot_hist == True:
            fig, ax = plt.subplots(len(runs), len(runs), figsize=(9,8))        
            # fig, ax = plt.subplots(len(runs), len(runs), figsize=(9*1.5,8*1.5))        
            runs_short = [i.split("max")[0] for i in runs]
            ticks = np.arange(0, lim+0.5, 0.5)
            for col in range(len(runs)-1,-1,-1):
                for row in range(len(runs)-1,-1,-1):
                    if col>row:
                        self.remove_frame(ax[row, col])
                        continue
                    if col==row:
                        df[runs[col]].hist(
                            ax=ax[row, col], 
                            bins=30,
                            range=(0,lim),
                            density=False,
                            color="tan",
                            edgecolor='black',
                            linewidth=0.3,
                            alpha=0.7
                            )
                        ax[row,col].set_xlabel(None)
                        ax[row,col].set_ylabel(None)
                        ax[row,col].grid(False)
                        ax[row,col].set_xlim([0,lim])
                        ax[row,col].text(s=labels[col], x=0.98,y=0.9, transform=ax[row,col].transAxes, ha="right")
                        
                        # ax[row,col].get_yaxis().set_ticks([])
                        ax[row, col].set_ylabel("Frequency", fontsize=8)
                        ax[row, col].set_xlabel("{} (m)" .format(stat), fontsize=8)
                        ax[row,col].tick_params(axis='y', labelsize=8)
                        ax[row,col].tick_params(axis='x', labelsize=8)
                        continue

                    # -- if get to this point, then making scatter plot
                    x_data = df[runs[col]]
                    y_data = df[runs[row]]

                    ax[row,col].scatter(x_data, y_data, s=10, facecolor="None", edgecolor='k')
                    ax[row,col].plot([-1,6], [-1,6], ls="-.", lw=1.0, zorder=1, color='r')
                    ax[row,col].set_xlim([0,lim])
                    ax[row,col].set_ylim([0,lim])

                    ax[row,col].tick_params(axis='x', labelsize=7)
                    ax[row,col].tick_params(axis='y', labelsize=7)

                    ax[row,col].set_xticks(ticks)
                    ax[row,col].set_yticks(ticks)

                    # -- 
                    ax[row,col].set_xlabel(labels[col], fontsize=8)
                    ax[row,col].set_ylabel(labels[row], fontsize=8)

                    # -- rmse and mae calcs
                    rmse = self.rmse(x_data, y_data)
                    mae  = self.mae(x_data, y_data)
                    s = "RMSE: {:0.3f}\nMAE:   {:0.3f}" .format(rmse, mae)
                    ax[row, col].text(x=0.05,y=0.95,s=s, transform=ax[row,col].transAxes, 
                                fontsize=8, va="top", ha="left",
                                bbox={"boxstyle":'square', "facecolor":'white', "alpha":0.5})
        else:
            fig, ax = plt.subplots(1,1, figsize=(4,3.5))
            ticks = np.arange(0, lim+0.5, 0.5)

            x_data = df[runs[0]]
            y_data = df[runs[1]]
            
            ax.scatter(x_data, y_data, s=10, facecolor="None", edgecolor='k')
            ax.plot([-1,6], [-1,6], ls="-.", lw=1.0, zorder=1, color='r')
            ax.set_xlim([0,lim])
            ax.set_ylim([0,lim])

            ax.tick_params(axis='x', labelsize=7)
            ax.tick_params(axis='y', labelsize=7)

            ax.set_xticks(ticks)
            ax.set_yticks(ticks)

            # -- 
            ax.set_xlabel(labels[0], fontsize=10)
            ax.set_ylabel(labels[1], fontsize=10)
            ax.set_title("Significant Wave Height at Buildings (m)", fontsize=10)
        
            rmse = self.rmse(x_data, y_data)
            mae  = self.mae(x_data, y_data)
            s = "RMSE: {:0.3f}\nMAE:   {:0.3f}" .format(rmse, mae)

            ax.text(x=0.05,y=0.85,s=s, transform=ax.transAxes, bbox={"boxstyle":'square', "facecolor":'white', "alpha":0.5})

            # plt.subplots_adjust(wspace=0.2, hspace=0.2)
            # plt.xticks(rotation=45)
        plt.tight_layout()
        self.save_fig(fig, fname, transparent=True, dpi=300)



    def scatter_domain(self, stat, runs, labels, plot_hist=True, lim=3, crude=False, fname=None):
        """ only use this when trying to compare runs with different size domains. 
            This code is so poorly written. not intended to be generalized, so be careful.
        """
        if crude:   
            runs.insert(0, self.model_runname)
            sizes = []
            for run in runs:
                H = self.read_npy(stat, run)
                sizes.append(np.shape(H)[0])
            largest_size_idx = np.argmax(sizes)

            wave_heights = {}
            for run in runs:
                H = self.read_npy(stat, run)                
                if run == "s1":
                    print("This is crudely done")
                    first_row = H[0:1, :]
                    last_row =  H[-1:, :]
                    H = np.vstack((first_row, H, last_row))
                    H = H[:,:-2]
                elif run == "s2":
                    print("This is crudely done")
                    first_row = H[0:1, :]
                    H = np.vstack((first_row, H))
                    H = H[:,:-1]

                wave_heights[run] = H

            H_domain = wave_heights[runs[largest_size_idx]]
            for run in runs:
                _, wave_heights[run] = self.check_domain_size_wave_stat(H_domain, wave_heights[run])

            df = pd.DataFrame()
            for run in runs:
                df[run] = wave_heights[run].flatten()
        else:
            df = pd.DataFrame()
            runs.insert(0, self.model_runname)
            sizes = []
            for run in runs:
                H = self.read_npy(stat, run)
                sizes.append(np.shape(H)[0])
            
            largest_size_idx = np.argmax(sizes)
            H_domain = self.read_npy(stat, runs[largest_size_idx])
            
            for run in runs:
                # read wave heights
                H = self.read_npy(stat, run)
                _, H = self.check_domain_size_wave_stat(H_domain, H)
                df[run] = H.flatten()
        # ---
        
        """ removing any row that is 0, this happens when comparing runs at
            different resolutions - wave heights at the buildings get funny
        """
        df = df[~(df == 0).any(axis=1)]

        # df = df.iloc[0:10000000]
        # print("temporarily using first 1,000,000 points")

        if plot_hist == True:
            fig, ax = plt.subplots(len(runs), len(runs), figsize=(9,8))
            # fig, ax = plt.subplots(len(runs), len(runs), figsize=(9*1.5,8*1.5))
            ticks = np.arange(0, lim+0.5, 0.5)
            for col in range(len(runs)-1,-1,-1):
                for row in range(len(runs)-1,-1,-1):
                    if col>row:
                        self.remove_frame(ax[row, col])
                        continue
                    if col==row:
                        df[runs[col]].hist(
                            ax=ax[row, col], 
                            bins=30,
                            range=(0,lim),
                            density=True,
                            color="tan",
                            edgecolor='black',
                            linewidth=0.3,
                            alpha=0.7
                            )
                        # df[runs[col]].plot.kde(
                        #     ax=ax[row,col],
                        #     color='k',
                        #     lw=1
                        #     )
                        ax[row,col].set_xlabel(None)
                        ax[row,col].set_ylabel(None)
                        ax[row,col].grid(False)
                        ax[row,col].set_xlim([0,lim])
                        ax[row,col].text(s=labels[col], x=0.98,y=0.9, transform=ax[row,col].transAxes, ha="right")
                        
                        ax[row, col].set_ylabel("Prob. Dens.", fontsize=8)
                        ax[row, col].set_xlabel("{} (m)" .format(stat), fontsize=8)
                        ax[row,col].tick_params(axis='x', labelsize=8)
                        ax[row,col].tick_params(axis='y', labelsize=8)
                        continue

                    # -- if get to this point, then making scatter plot
                    x_data = df[runs[col]]
                    y_data = df[runs[row]]

                    ax[row,col].scatter(x_data, y_data, s=0.001, lw=0.01, alpha=0.9, marker=".", facecolor="None", edgecolor='k')
                    ax[row,col].plot([-1,6], [-1,6], ls="-.", lw=1.0, zorder=1, color='r')
                    ax[row,col].set_xlim([0,lim])
                    ax[row,col].set_ylim([0,lim])

                    ax[row,col].tick_params(axis='x', labelsize=7)
                    ax[row,col].tick_params(axis='y', labelsize=7)

                    ax[row,col].set_xticks(ticks)
                    ax[row,col].set_yticks(ticks)

                    # -- 
                    ax[row,col].set_xlabel(labels[col], fontsize=8)
                    ax[row,col].set_ylabel(labels[row], fontsize=8)
                    # --


                    rmse = self.rmse(x_data, y_data)
                    mae  = self.mae(x_data, y_data)
                    s = "RMSE: {:0.3f}\nMAE:   {:0.3f}" .format(rmse, mae)

                    ax[row, col].text(x=0.05,y=0.8,s=s, transform=ax[row,col].transAxes, fontsize=8,
                                bbox={"boxstyle":'square', "facecolor":'white', "alpha":0.5})
            # plt.subplots_adjust(wspace=0.2, hspace=0.2)
        else:
            fig, ax = plt.subplots(1,1, figsize=(4,3.5))
            ticks = np.arange(0, lim+0.5, 0.5)
            
            x_data = df[runs[0]]
            y_data = df[runs[1]]

            ax.scatter(x_data, y_data, s=10, facecolor="None", edgecolor='k', zorder=0)
            ax.plot([-1,6], [-1,6], ls="-.", lw=1.0, color='r', zorder=1)
            ax.set_xlim([0,lim])
            ax.set_ylim([0,lim])

            ax.tick_params(axis='x', labelsize=7)
            ax.tick_params(axis='y', labelsize=7)

            ax.set_xticks(ticks)
            ax.set_yticks(ticks)

            # -- 
            ax.set_xlabel(labels[0], fontsize=10)
            ax.set_ylabel(labels[1], fontsize=10)
            ax.set_title("Significant Wave Over Domain (m)", fontsize=10)    
            
            rmse = self.rmse(x_data, y_data)
            mae  = self.mae(x_data, y_data)
            s = "RMSE: {:0.3f}\nMAE:   {:0.3f}" .format(rmse, mae)

            ax.text(x=0.05,y=0.85,s=s, transform=ax.transAxes, bbox={"boxstyle":'square', "facecolor":'white', "alpha":0.5})

        plt.tight_layout()
        self.save_fig(fig, fname, transparent=True, dpi=300)





















