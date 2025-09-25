import os
import numpy as np
import pandas as pd
import scipy.stats as st
import scipy.ndimage as ndi
import scipy.signal as sg
import xarray as xr
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as patches


class plot_wave_heights():
    """docstring for plot_wave_heights"""
    def __init__(self, var="H", stat="max"):
        self.file_dir = os.path.dirname(os.path.realpath(__file__))
        self.path_to_model = os.path.join(self.file_dir, "..","..", "xbeach", "models")
        self.var = var
        self.stat = stat


    def get_output_filename(self, fn):
        # fn = os.path.join(self.path_to_model, model_runname)
        files = os.listdir(fn)
        fn  = [i for i in files if "xboutput" in i][0]
        return fn

    def plot_wave_height_bldg(self, model_runname, model_runname_w_bldgs=None, 
                                readlocal=False, vmax=1, vmin=0, domain_size="estero", 
                                grey_background=False, fname=None):
        # read wave heights
        H, model_runname = self.read_local_or_ncdf(model_runname, readlocal)
        
        # read buildings and grid 
        if model_runname_w_bldgs != None:
            # H_, model_runname_w_bldgs = self.read_local_or_ncdf(model_runname_w_bldgs, readlocal)
            bldgs, xgr, ygr, zgr = self.read_bldgs_grd(model_runname_w_bldgs)
        else:
            bldgs, xgr, ygr, zgr = self.read_bldgs_grd(model_runname)
        
        # assign max H to each building
        bldg_H = self.assign_max_to_bldgs(H, bldgs)

        # setting up cmap
        cmap = mpl.cm.plasma
        if grey_background:
            cmap.set_bad("grey")
        else:
            cmap.set_bad(alpha=0)

        figsize = self.get_figsize(domain_size)        
        fig, ax = plt.subplots(1,1, figsize=figsize)
        ax.pcolormesh(xgr, ygr, zgr, vmin=-8.5, vmax=8.5, cmap="BrBG_r", zorder=0)
        pcm = ax.pcolormesh(xgr, ygr, bldg_H, vmin=vmin, vmax=vmax, cmap=cmap, zorder=1)
        plt.colorbar(pcm, ax=ax, extend="max", label="Max Wave Height (m)", aspect=40)
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")

        self.savefig(fname)

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
            H, model_runname = self.read_local_or_ncdf(run, readlocal)

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
        self.savefig(fname)
    
    def print_summary_statistics(self, runs, model_runname_w_bldgs=None, readlocal=True):
        if model_runname_w_bldgs != None:
            # first getting mask where buildings are
            H_, model_runname_w_bldgs = self.read_local_or_ncdf(model_runname_w_bldgs, readlocal)
            _, mask_bldgs, xgr, ygr, zgr = self.read_bldgs_grd(model_runname_w_bldgs, readlocal, H_)

        df = pd.DataFrame()
        for run in runs:
            # read wave heights
            H, model_runname = self.read_local_or_ncdf(run, readlocal)
            
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

    def plot_wave_height_domain(self, model_runname, readlocal=False, vmax=None, vmin=None, 
                             fname=None, prnt_read=False, single_frame=False,
                             domain_size="estero", plt_bldgs=True):
        
        # read wave heights
        H, model_runname = self.read_local_or_ncdf(model_runname, readlocal)

        # read buildings and grid 
        bldgs, xgr, ygr, zgr = self.read_bldgs_grd(model_runname)
        mask = np.ma.getmask(bldgs)
        figsize = self.get_figsize(domain_size)

        # fig, ax = plt.subplots(1,1, figsize=figsize)
        if single_frame:
            fig, ax0 = plt.subplots(1,1, figsize=figsize)
        else:
            fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(16,9), gridspec_kw={'width_ratios': [1,2.8]})

        # setting up mask to ignore values less than 0
        mask = (zgr<=0)
        masked_array = np.ma.array(H, mask=mask)

        # setting up colormap for water
        if self.var == "H":
            cmap = mpl.cm.plasma
            cmap.set_bad('grey')
            vmax = 1.0 if vmax == None else vmax
            vmin = 0
        else:
            cmap = mpl.cm.plasma
            # cmap = mpl.cm.cividis
            cmap.set_bad('grey')
            vmax = 3.0 if vmax == None else vmax
            vmin = 0.0 if vmin == None else vmin

        # -- drawing first plot
        pcm = ax0.pcolormesh(xgr, ygr, masked_array, vmin=vmin, vmax=vmax, cmap=cmap)
        if single_frame:
            ax_bar = ax0
        else:
            ax_bar = ax1
        plt.colorbar(pcm, ax=ax_bar, extend="max", label="Sig. Wave Height (m)", aspect=40)
        if plt_bldgs:
            custom_color = 'springgreen'
            cmap_bldg = mpl.colors.ListedColormap([custom_color])
            cmap_bldg.set_bad(alpha=0)

            ax0.pcolormesh(xgr, ygr, bldgs, cmap=cmap_bldg)
        ax0.set_xlabel("x (m)")
        ax0.set_ylabel("y (m)")
        # ax0.set_title(s)
        # self.remove_frame(ax0)

        if single_frame==False:
            # -- drawing second, zoomed in plot
            # full model domain
            box_lower_left = (2600, 5000)       # in world units
            dx, dy = 1000, 1000
            # continuing with zommed in plot
            box_upper_right = (box_lower_left[0]+dx, box_lower_left[1]+dy)

            id_ll = self.wrld_to_grid_index(xgr, ygr, box_lower_left)
            id_ur = self.wrld_to_grid_index(xgr, ygr, box_upper_right)
            
            xgr2 = xgr[id_ll[1]:id_ur[1], id_ll[0]:id_ur[0]]
            ygr2 = ygr[id_ll[1]:id_ur[1], id_ll[0]:id_ur[0]]
            masked_array2 = masked_array[id_ll[1]:id_ur[1], id_ll[0]:id_ur[0]]
            bldgs2 = bldgs[id_ll[1]:id_ur[1], id_ll[0]:id_ur[0]]

            ax1.pcolormesh(xgr2, ygr2, masked_array2, vmin=vmin, vmax=vmax, cmap=cmap)
            ax1.pcolormesh(xgr2, ygr2, bldgs2, cmap=cmap_bldg)

            # # --old
            box_l = xgr2[0,-1] - xgr2[0,0]
            box_h = ygr2[-1,0] - ygr2[0,0]

            # # -- adding rectangle showing where zoomed in area is
            rect = patches.Rectangle(box_lower_left, box_l, box_h, linewidth=3, zorder=10, edgecolor='r', facecolor='none')
            ax0.add_patch(rect)

        # --- saving file
        self.savefig(fname)


    def plot_wave_height_hist(self, runs,read_max_local=False, fname=None):
        
        lbls = self.run2label_dict()
        df = pd.DataFrame()
        runname_list = []
        cnt = 0
        for run in runs:
            # read max wave heights
            run_max, model_runname = self.read_local_or_ncdf(run, read_max_local)
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
        self.savefig(fname)
       

    def plot_wave_height_scatter(self, runs, readlocal=True, plt_same_axis=True, fname=None):
        df = pd.DataFrame()
        for run in runs:
            # read wave heights
            H, model_runname = self.read_local_or_ncdf(run, readlocal)
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
        self.savefig(fname)

    def plot_wave_height_domain_diff(self, run1, run2, r1local=False, r2local=False, 
                                  vmax=1, norm=False, domain_size="estero",
                                  fname=None):
        # get difference in wave heights
        run1_max, run1 = self.read_local_or_ncdf(run1, r1local)
        run2_max, run2 = self.read_local_or_ncdf(run2, r2local)

        mask = np.isnan(run1_max) & np.isnan(run2_max)
        run1_max = np.ma.array(run1_max, mask=mask) # here mask tells numpy which cells to ignore.
        run2_max = np.ma.array(run2_max, mask=mask) # here mask tells numpy which cells to ignore.

        if norm == True:
            # max_max = np.maximum(run1_max, run2_max)
            denom = (run1_max+run2_max)/2
            diff = ((run1_max - run2_max)/denom)
        else:
            diff = run1_max - run2_max

        # -- read in buildngs
        if r1local == False:
            # read buildings
            model_dir = os.path.join(self.path_to_model, model_runname)
            bldgs, mask = self.read_buildings(model_dir, rtn_mask=True)
        else:
            model_runs = os.listdir(self.path_to_model)
            model_run = [i for i in model_runs if run1.split("max")[0] in i]
            model_run = [i for i in model_run if ".tar.gz" not in i][0]        
            model_dir = os.path.join(self.path_to_model, model_run)
            mask = ~np.isnan(run2_max)
            bldgs = np.ma.array(run2_max, mask=mask)

        # -- read in grid
        xgr, ygr, zgr = self.read_grid(model_dir)
        figsize = self.get_figsize(domain_size)

        fig, ax0 = plt.subplots(1,1, figsize=figsize)

        # determine where water is and setup mask to ignore those cells.
        mask = (zgr<0)
        masked_array = np.ma.array(diff, mask=mask) # here mask tells numpy which cells to ignore.
        
        # setting up colormap for water
        if self.var == "H":
            cmap = mpl.cm.RdBu
            cmap = mpl.cm.BrBG
            cmap.set_bad('grey')
            vmax = 1.0 if vmax == None else vmax
            vmin = -vmax
        else:
            cmap = mpl.cm.cividis
            cmap.set_bad('bisque')
            vmax = 3.0 if vmax == None else vmax
            vmin = 0.0

        cmap_bldg = mpl.cm.Greys_r
        cmap_bldg.set_bad(alpha=0)

        # -- drawing first plot
        if norm:
            label = "Percent Difference"
            vmax = 1
            vmin = -1
            extend = None
        else:
            label = "Difference (m)"
            extend = "both"
        pcm = ax0.pcolormesh(xgr, ygr, masked_array, vmin=vmin, vmax=vmax, cmap=cmap)
        plt.colorbar(pcm, ax=ax0, extend=extend, label=label, aspect=40)
        ax0.pcolormesh(xgr, ygr, bldgs, cmap=cmap_bldg)
        ax0.set_xlabel("x (m)")
        ax0.set_ylabel("y (m)")

        # --- saving file
        self.savefig(fname)

    def read_buildings(self, model_dir, rtn_mask=False):
        fn_zgrid = os.path.join(model_dir, "z.grd")
        zs = []
        with open(fn_zgrid,'r') as f:
            for cnt, line in enumerate(f.readlines()):
                z_ = [float(i.strip()) for i in line.split()]
                zs.append(z_)
        zgr = np.array(zs)
        mask = (zgr != 10)
        bldgs = np.ma.array(zgr, mask=mask)
        if rtn_mask:
            return bldgs, mask
        return bldgs

    def get_H(self, z, detrend=False):
        if detrend:
            z = z - np.mean(z)  # de-trend signal with mean
        
        # The sign of the (detrended) elevation at each point
        signs = np.sign(z)
        # Find where the sign changes.
        zero_crossing_indices = np.where(np.diff(signs) != 0)[0]

        up_crossings = zero_crossing_indices[np.where(signs[zero_crossing_indices] < signs[zero_crossing_indices + 1])[0]]
        # Ensure we have pairs of up-crossings to define full waves
        start_indices = up_crossings[:-1]
        end_indices = up_crossings[1:]

        # Use a list comprehension to get max and min values for each segment
        crests = [np.max(z[start:end]) for start, end in zip(start_indices, end_indices)]
        troughs = [np.min(z[start:end]) for start, end in zip(start_indices, end_indices)]

        # Convert to NumPy arrays for vectorized subtraction
        wave_heights = np.array(crests) - np.array(troughs)
        return wave_heights

    def compute_Hs(self, H):
        H_one_third = np.quantile(H, q=2/3)
        H = H[H>H_one_third]
        Hs = np.mean(H)
        return Hs

    def save_wave_stats(self, model_runname, stat):
        fn_out = "{}_{}" .format(model_runname, stat)
        fn = os.path.join(self.path_to_model, model_runname)

        dims = self.read_data_xarray_dims(fn)
        data_save = np.empty(dims)
        data_all = self.read_data_xarray_test(fn, self.var)
        for y2_ in range(dims[0]):
            print("y2_ = {} out of {}" .format(y2_, dims[0]))
            for x2_ in range(dims[1]):
                z = data_all[:,y2_,x2_]
                if np.sum(z) == 0:
                    data_ = 0
                else:
                    H = self.get_H(z)       # getting wave heights from time series
                    if len(H) == 0:
                        data_ = 0
                    elif stat == "Hmax":
                        data_ = np.max(H)
                    elif stat == "Hs":
                        data_ = self.compute_Hs(H)
                data_save[y2_, x2_] = data_

        fn_out = os.path.join(fn, fn_out)
        np.save(fn_out, data_save)
        print("max wave height saved as: {}" .format(fn_out+".npy"))

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

    def assign_max_to_bldgs(self, data, bldgs):
        max_H = np.empty(np.shape(data))
        max_H[:] = np.nan
        mask = np.ma.getmask(bldgs)
        labeled_mask, num_features = ndi.label(~mask)
        for i in range(num_features+1):
            if i == 0:
                continue
            m_ = labeled_mask==i
            m_ = ~m_
            m_ = np.pad(m_, pad_width=1, mode="constant", constant_values=True)

            shifted_up = m_[2:, 1:-1]
            shifted_down = m_[:-2, 1:-1]
            shifted_left = m_[1:-1, 2:]
            shifted_right = m_[1:-1, :-2]
            original_mask_trimmed = m_[1:-1, 1:-1]

            offset_mask = original_mask_trimmed & shifted_up & shifted_down & shifted_left & shifted_right
            offset_mask = ~offset_mask

            max_H[labeled_mask==i] = np.nanmax(data[offset_mask])

        return max_H

    def read_bldgs_grd(self, model_runname):
        model_dir = os.path.join(self.path_to_model, model_runname)
        xgr, ygr, zgr = self.read_grid(model_dir)
        mask = (zgr != 10)
        bldgs = np.ma.array(zgr, mask=mask)
        
        return bldgs, xgr, ygr, zgr


        # --------- old
        # if readlocal == False:
        #     # read buildings
        #     model_dir = os.path.join(self.path_to_model, model_runname)
        #     bldgs, mask = self.read_buildings(model_dir, rtn_mask=True)
        # else:
        #     model_runs = os.listdir(self.path_to_model)
        #     model_run = [i for i in model_runs if model_runname.split("max")[0] in i]
        #     model_run = [i for i in model_run if ".tar.gz" not in i][0]        
        #     model_dir = os.path.join(self.path_to_model, model_run)
        #     mask = ~np.isnan(H)
        #     bldgs = np.ma.array(H, mask=mask)

        # # -- read grid
        # xgr, ygr, zgr = self.read_grid(model_dir)

        # return bldgs, mask, xgr, ygr, zgr

    def savefig(self, fname):
        if fname!=None:
            plt.savefig(fname,
                        transparent=True,
                        dpi=300,
                        bbox_inches='tight',
                        pad_inches=0.1,
                        )
            plt.close()

    def get_figsize(self, domain_size):
        if domain_size=="micro":
            figsize=(7,5)
        else:
            figsize=(3,8)
        return figsize

    def read_local_or_ncdf(self, run, rlocal):
        if rlocal:
            runname = run.split("_")[0]
            all_runs = os.listdir(self.path_to_model)
            runname = [i for i in all_runs if runname in i]
            runname = [i for i in runname if ".tar.gz" not in i][0]
            fn = os.path.join(self.path_to_model, runname, run)
            rmax = np.load(fn)
            
        else:
            fn = os.path.join(self.path_to_model, run)
            rmax = self.read_data_xarray_max(fn, var="H")
            runname = run
        return rmax, runname

    def wrld_to_grid_index(self, xgr, ygr, xy):
        idx = np.argmin(np.abs(xgr[0,:] - xy[0]))
        idy = np.argmin(np.abs(ygr[:,0] - xy[1]))        
        return (idx,idy)

    def remove_frame(self, ax):
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.get_xaxis().set_ticks([])
        ax.get_yaxis().set_ticks([])

    def read_data_xarray_dims(self, model_dir):
        output_filename = self.get_output_filename(model_dir)
        fn = os.path.join(model_dir, output_filename)
        ds = xr.open_dataset(fn, chunks={"globaltime": 100})
        
        nx = ds.sizes["nx"]
        ny = ds.sizes["ny"]
        return (ny, nx)

    def read_data_xarray_test(self, model_dir, var):
        output_filename = self.get_output_filename(model_dir)
        fn = os.path.join(model_dir, output_filename)
        ds = xr.open_dataset(fn, chunks={"globaltime": 100})
        return ds[var][:,:,:].values

    def read_data_xarray_pt(self, model_dir, var, idx, idy):
        output_filename = self.get_output_filename(model_dir)
        fn = os.path.join(model_dir, output_filename)
        ds = xr.open_dataset(fn, chunks={"globaltime": 100})
        slice_data = ds[var][:,idy,idx]
        return slice_data.values

    def read_data_xarray_max(self, model_dir, var, prnt_read=False):
        output_filename = self.get_output_filename(model_dir)
        fn = os.path.join(model_dir, output_filename)
        ds = xr.open_dataset(fn, chunks={"globaltime": 100})
        
        if prnt_read:
            print("Dataset object read:")
            print(ds)
            print("\n\n")
        
        max_vals = ds[var].max(dim="globaltime").values[:,:]
        return max_vals
    
    def read_data_xarray(self, model_dir, var, t, prnt_read=False, rtn_time_array=False):
        output_filename = self.get_output_filename(model_dir)
        fn = os.path.join(model_dir, output_filename)
        ds = xr.open_dataset(fn, chunks={"globaltime": 100})
        if prnt_read:
            print("Dataset object read:")
            print(ds)
            print("\n\n")
        
        slice_data = ds[var].isel(globaltime=slice(t,t+1))
        if rtn_time_array:
            time = ds["globaltime"].values
            # print("Last time step: {} hr." .format(time[-1]/60/60))
            return slice_data.values[0,:,:], time
        else:
            return slice_data.values[0,:,:]
    
    def read_grid(self, model_dir):
        fn_xgrid = os.path.join(model_dir, "x.grd")
        if os.path.isfile(fn_xgrid):
            xgrid = os.path.join(model_dir, "x.grd")
            ygrid = os.path.join(model_dir, "y.grd")
            zgrid = os.path.join(model_dir, "z.grd")

            with open(xgrid,'r') as f:
                for cnt, line in enumerate(f.readlines()):
                    xs = [float(i.strip()) for i in line.split()]
                    if cnt == 0:
                        break
            ys = []
            with open(ygrid,'r') as f:
                for cnt, line in enumerate(f.readlines()):
                    y_ = [float(i.strip()) for i in line.split()][0]
                    ys.append(y_)
            
            zgr = np.zeros((len(ys), len(xs)))
            with open(zgrid,'r') as f:
                for cnt, line in enumerate(f.readlines()):
                    z_ = [float(i.strip()) for i in line.split()]
                    zgr[cnt,:] = z_

            xgr, ygr = np.meshgrid(xs, ys)
        else:
            fn_params = os.path.join(model_dir, "params.txt")
            with open(fn_params) as f:
                for cnt, line in enumerate(f.readlines()):
                    ls = [i.strip() for i in line.split()]
                    if "dx" in ls:
                        dx = float(ls[-1])
                    elif "dy" in ls:
                        dy = float(ls[-1])
                    elif "nx" in ls:
                        nx = float(ls[-1])
                    elif "ny" in ls:
                        ny = float(ls[-1])
            
            xs = np.arange(start=0, stop=nx*dx+dx, step=dx)
            ys = np.arange(start=0, stop=ny*dy+dx, step=dy)
            xgr, ygr = np.meshgrid(xs, ys)

        return xgr, ygr, zgr


if __name__ == "__main__":
    
    pwh = plot_wave_heights(var="zs1")
    pwh.save_wave_stats("run26", stat="Hs")

    # pwh.plot_wave_height_domain(model_runname="run27-copy_Hs.npy", 
    #                          readlocal=True,
    #                          vmin=0,
    #                          vmax=1,
    #                          single_frame=True, 
    #                          domain_size="micro",
    #                          plt_bldgs=True,
    #                          fname="run27-copy_Hs.png"
    #                          )

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

    pwh.plot_wave_height_scatter(runs=["run"],
                                readlocal=True,
                                plt_same_axis=True
                                 # fname="temp.png"
                                 )

    # pwh.plot_wave_height_hist(
    #                        runs = ["run15max.npy", "run16max.npy", "run18max.npy"],
    #                        read_max_local=True, 
    #                        fname="r15r16r18_hist.png"
    #                        )



    plt.show()

