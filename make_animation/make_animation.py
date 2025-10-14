import os
import shutil
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import imageio
import seaborn as sns
import xarray as xr
import multiprocessing

from helpers.helpers import HelperFuncs


class MakeAnimation(HelperFuncs):
    """docstring for xb_plotting_large"""
    def __init__(self, var="H", tstart=None, 
                tstop=None, domain_size="estero", xbeach_duration=12, vmax=1, 
                vmin=0, make_all_figs=True, dpi=300):
        super().__init__()

        self.file_dir = os.path.dirname(os.path.realpath(__file__))
        # self.path_to_model = path_to_model
        # self.xboutput_filename = self.get_output_filename()
        # self.model_runname = self.path_to_model.split(os.sep)[-1]

        self.var = var
        self.tstart = tstart
        self.tstop = tstop
        self.domain_size = domain_size
        self.xbeach_duration = xbeach_duration
        self.vmax = vmax
        self.vmin = vmin
        self.make_all_figs = make_all_figs
        self.dpi = dpi

    def plot_timestep_micro(self, t_hr=None, fname=None, t_start=None, t_stop=None):
        if t_hr!=None:
            t = self.read_time_xarray()
            t_idx = np.argmin(np.abs(t-t_hr*3600))
        else:
            t_idx = -1
        
        xgr, ygr, _ = self.read_grid()                                     # reading grid data
        data_plot = self.read_2d_data_xarray_timestep(var=self.var, t=t_idx)        # reading xbeach output
        mask = (data_plot < -99999)
        masked_array = np.ma.array(data_plot, mask=mask)

        # setting some info for the plot        
        cmap, cmap_bldg = self.setup_coloramp()
        s, cbar_s = self.get_labels(t, t_idx)

        # -- make figure and subplots
        figsize = (10,8)
        fig, (ax0, ax1) = plt.subplots(2,1, figsize=figsize, height_ratios=[8,1])

        # -- drawing first plot
        bldgs = self.read_buildings()
        pcm = ax0.pcolormesh(xgr, ygr, masked_array, vmin=self.vmin, vmax=self.vmax, cmap=cmap)
        plt.colorbar(pcm, ax=ax0, extend="both", label=cbar_s, aspect=40)
        ax0.pcolormesh(xgr, ygr, bldgs, cmap=cmap_bldg)
        ax0.set_title(s)

        self.draw_time_series(ax1, t_hr, t_start, t_stop)

        # --- saving figure
        self.save_fig(fig, 
                    fname, 
                    transparent=False, 
                    dpi=self.dpi,
                    )

    def plot_timestep(self, t_hr, fname=None, prnt_read=False, t_start=None, t_stop=None):
        """ function to plot single timestep
        """

        t = self.read_time_xarray()
        t_idx = np.argmin(np.abs(t-t_hr*3600))
        t_hr = t[t_idx]/3600


        data_plot = self.read_2d_data_xarray_timestep(var=self.var, t=t_idx)
        xgr, ygr, _ = self.read_grid()

        figsize = (14,9)
        fig = plt.figure(figsize=figsize)
        gs = mpl.gridspec.GridSpec(2, 2, figure=fig, width_ratios=[1, 4], height_ratios=[4, 1])
        ax0 = fig.add_subplot(gs[0:, 0])
        ax1 = fig.add_subplot(gs[0, 1])
        ax2 = fig.add_subplot(gs[1, 1])

        # --- new
        mask = (data_plot < -99999)
        masked_array = np.ma.array(data_plot, mask=mask)
        
        cmap, cmap_bldg = self.setup_coloramp()
        s, cbar_s = self.get_labels(t, t_idx)

        # -- drawing first plot
        pcm = ax0.pcolormesh(xgr, ygr, masked_array, vmin=self.vmin, vmax=self.vmax, cmap=cmap)
        plt.colorbar(pcm, ax=ax1, extend="both", label=cbar_s)

        bldgs = self.read_buildings()
        ax0.pcolormesh(xgr, ygr, bldgs, cmap=cmap_bldg)
        ax0.set_title(s)

        # -- drawing second, zoomed in plot
        # full model domain
        box_lower_left = (1600, 5000)       # in world units
        dx, dy = 1000, 1000

        # continuing with zommed in plot
        box_upper_right = (box_lower_left[0]+dx, box_lower_left[1]+dy)

        id_ll = self.xy_to_grid_index(xgr, ygr, box_lower_left)
        id_ur = self.xy_to_grid_index(xgr, ygr, box_upper_right)
        
        xgr2 = xgr[id_ll[1]:id_ur[1], id_ll[0]:id_ur[0]]
        ygr2 = ygr[id_ll[1]:id_ur[1], id_ll[0]:id_ur[0]]
        masked_array2 = masked_array[id_ll[1]:id_ur[1], id_ll[0]:id_ur[0]]
        bldgs2 = bldgs[id_ll[1]:id_ur[1], id_ll[0]:id_ur[0]]
        
        ax1.pcolormesh(xgr2, ygr2, masked_array2, vmin=self.vmin, vmax=self.vmax, cmap=cmap)
        ax1.pcolormesh(xgr2, ygr2, bldgs2, cmap=cmap_bldg)
        
        box_l = xgr2[0,-1] - xgr2[0,0]
        box_h = ygr2[-1,0] - ygr2[0,0]

        # -- adding rectangle showing where zoomed in area is
        rect = patches.Rectangle(box_lower_left, box_l, box_h, linewidth=3, zorder=10, edgecolor='r', facecolor='none')
        ax0.add_patch(rect)
        
        self.draw_time_series(ax2, t_hr, t_start, t_stop)


        # -- (optionally) saving file
        self.save_fig( fig, 
                    fname, 
                    transparent=True, 
                    dpi=self.dpi, 
                    )

    def draw_time_series(self, ax, t_hr, t_start, t_stop):
        # # -- now plotting time series
        forcing = self.frcing_to_dataframe()
        x,y = forcing["t_hr"], forcing["el"]
        ax.plot(x,y, lw=0.75, ls="-.", color='k', zorder=0, label="ADCIRC/SWAN")
        
        start_idx = forcing.loc[forcing["t_sec"]==t_start*3600].index[0]
        stop_idx = forcing.loc[forcing["t_sec"]==t_stop*3600].index[0]

        df_trnc = forcing.iloc[start_idx:stop_idx]
        ax.plot(df_trnc["t_hr"], df_trnc["el"], color="#ff5370", lw=2, zorder=1, label="XBeach")

        df_trnc.reset_index(inplace=True)
        t_trnc_hr = df_trnc["t_hr"].iloc[0]+t_hr

        # interpolating water level on time series
        y_ = np.interp(t_trnc_hr, df_trnc["t_hr"].values, df_trnc["el"].values)
        ax.scatter(t_trnc_hr, y_, color="k", s=40, zorder=2)
        ax.set_xlabel("Time (hr)")
        ax.set_ylabel("Water Elevation (m)")
        ax.set_xlim([20,90])
        leg = ax.legend(loc="upper left", facecolor='white')
        leg.get_frame().set_alpha(None)

    def get_labels(self, time, t_idx):
        if self.var == "H":
            s = "Time: {:2.1f}h ({:8.0f}s)" .format(time[t_idx]/3600, time[t_idx])
            cbar_s = "Wave Height (m)"
        elif self.var == "zs":
            s = "Time: {:2.1f}h ({:8.0f}s)" .format(time[t_idx]/3600, time[t_idx])
            cbar_s = "Water Elevation (m)"
        elif self.var == "zs0":
            s = "Time {:2.1f}h ({:8.0f}s)" .format(time[t_idx]/3600, time[t_idx])
            cbar_s = "Water Elevation - Tide Alone (m)"
        elif self.var == "zs1":
            s = "Time {:2.1f}h ({:8.0f}s)" .format(time[t_idx]/3600, time[t_idx])
            cbar_s = "Water Elevation - Minus Tide (m)"
        return s, cbar_s

    def setup_coloramp(self):
        # setting up colormap for water
        if self.var == "H":
            cmap = mpl.cm.plasma
            cmap.set_bad('bisque')
        else:
            cmap = mpl.cm.plasma
            # cmap = mpl.cm.Blues_r
            # cmap = mpl.cm.berlin
            cmap.set_bad('bisque')

        # setting color for buildings.
        custom_color = 'springgreen'
        cmap_bldg = mpl.colors.ListedColormap([custom_color])
        cmap_bldg.set_bad(alpha=0)

        return cmap, cmap_bldg

    def make_animation(self, parallel=True, num_proc=None):
        t = self.read_time_xarray()
        if self.tstart == None:
            self.tstart = t[0]
        if self.tstop == None:
            self.tstop = t[-1]/3600
        tstart_idx = np.argmin(np.abs(t-self.tstart*3600))
        tstop_idx = np.argmin(np.abs(t-self.tstop*3600))

        t_start_xbeach, t_stop_xbeach = self.xbeach_duration_to_start_stop()
        print("creating video with tstart = {:.2f} hr and tstop = {:.2f} hr" .format(self.tstart, self.tstop))
        print("  found nearest time steps as: tstart = {:.2f} hr and tstop = {:.2f}hr" .format(t[tstart_idx]/3600, t[tstop_idx]/3600))
        print("  making video with time indices: tstart_idx = {} and tstop_idx = {}" .format(tstart_idx, tstop_idx))
        # --- making images to comprise video
        temp_dir = os.path.join(self.file_dir, "temp")
        if self.make_all_figs:
            if os.path.isdir(temp_dir):
                shutil.rmtree(temp_dir)
            self.make_directory(temp_dir)
            if parallel:
                my_list = []
                for t_ in range(tstart_idx, tstop_idx):
                    my_list.append((t_, t, temp_dir, t_start_xbeach, t_stop_xbeach))

                with multiprocessing.Pool(num_proc) as pool:
                    pool.starmap(self.make_frame, my_list)

            else: # series 
                for t_ in range(tstart_idx, tstop_idx):
                    if t_%10==0:
                        print(t_)
                    self.make_frame(t_, t ,temp_dir, t_start_xbeach, t_stop_xbeach)
                    plt.close()

        if self.domain_size=="micro":
            figsize = (10,8)
        else:
            figsize = (16,9)
        self.matplotlib_writer(tstart_idx, tstop_idx, temp_dir, figsize)

    def make_frame(self, t_, t, temp_dir, t_start_xbeach, t_stop_xbeach):
        fn = os.path.join(temp_dir, "f{}.png" .format(t_))
        t_hr = t[t_]/3600
        if self.domain_size == "estero":
            self.plot_timestep(t_hr=t_hr, fname=fn, t_start=t_start_xbeach, t_stop=t_stop_xbeach)
        elif self.domain_size == "micro":
            self.plot_timestep_micro(t_hr=t_hr, fname=fn, t_start=t_start_xbeach, t_stop=t_stop_xbeach)
        plt.close()

    def plot_frame(self, t_hr):
        t = self.read_time_xarray()
        t_idx = np.argmin(np.abs(t-t_hr*3600))
        t_hr_plot = (t[t_idx])/3600
        t_start_xbeach, t_stop_xbeach = self.xbeach_duration_to_start_stop()
        print("creating frame at t_hr = {:.2f} hr" .format(t_hr_plot))
        print("  nearest time step to input <{:.2f}> hr is t_hr = {:.2f} hr" .format(t_hr, t_hr_plot))
        print("  this corresponds to time array index: t_idx={}" .format(t_idx))

        fn = "f{}.png" .format(t_idx)
        
        if self.domain_size == "estero":
            self.plot_timestep(t_hr=t_hr_plot, fname=fn, t_start=t_start_xbeach, t_stop=t_stop_xbeach)
        elif self.domain_size == "micro":
            self.plot_timestep_micro(t_hr=t_hr_plot, fname=fn, t_start=t_start_xbeach, t_stop=t_stop_xbeach)

    def matplotlib_writer(self, tstart_idx, tstop_idx, temp_dir, figsize):
        video_name = '{}-{}.mp4'.format(self.model_runname, self.var)
        video_name = os.path.join(self.path_to_save_plot, video_name)
        fig, ax = plt.subplots(figsize=figsize)
        writer = animation.FFMpegWriter(fps=10)
        with writer.saving(fig, video_name, dpi=self.dpi):
          for step in range(tstart_idx, tstop_idx):
                if step%100==0:
                    print("making video at step: {}" .format(step))
                fn = os.path.join(temp_dir, "f{}.png".format(step))
                if os.path.isfile(fn):
                    image = plt.imread(fn)
                    ax.clear()
                    ax.imshow(image)
                    ax.axis('off')  # Optional: Hide axes for a cleaner look
                    plt.tight_layout()
                    writer.grab_frame()
        plt.close()
        # Clean up the temporary directory
        if os.path.isdir(temp_dir):
            shutil.rmtree(temp_dir)


