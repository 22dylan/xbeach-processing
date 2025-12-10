import os
import shutil
import numpy as np
import pandas as pd

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import multiprocessing

from helpers.helpers import HelperFuncs



class MakeAnimationHotstart(HelperFuncs):
    """docstring for xb_plotting_large"""
    def __init__(self, var="H", tstart=None, 
                tstop=None, domain_size="estero", xbeach_duration=12, vmax=1, 
                vmin=0, make_all_figs=True, dpi=300, fps=10, detrend=False, dt_video=1, 
                hotstart_runs=None):
        super().__init__()

        self.file_dir = os.path.dirname(os.path.realpath(__file__))
        
        self.var = var
        self.tstart = tstart
        self.tstop = tstop
        self.domain_size = domain_size
        self.xbeach_duration = xbeach_duration
        self.vmax = vmax
        self.vmin = vmin
        self.make_all_figs = make_all_figs
        self.dpi = dpi
        self.fps = fps
        self.detrend = detrend
        self.dt_video = dt_video
        self.set_hotstart_runs()


    def make_animation_hotstart(self, parallel=False, num_proc=None):
        t_df = self.construct_time_df(self.hotstart_runs)
        
        t_start_row = t_df.loc[(t_df["t_hr"] - self.tstart).abs().idxmin()]
        t_stop_row  = t_df.loc[(t_df["t_hr"] - self.tstop).abs().idxmin()]

        print("creating video with tstart = {:.2f} hr and tstop = {:.2f} hr" .format(self.tstart, self.tstop))
        print("  found nearest time steps as: tstart = {:.2f} hr and tstop = {:.2f}hr" .format(t_start_row["t_hr"], t_stop_row["t_hr"]))
        print("  making video with time indices: tstart_idx = {} and tstop_idx = {}" .format(t_start_row["t_idx"], t_stop_row["t_idx"]))
        print("  hotstart runs will span: {} to {}" .format(t_start_row["run"], t_stop_row["run"]))
        # --- making images to comprise video
        temp_dir = os.path.join(self.file_dir, "temp")
        if self.make_all_figs:
            if os.path.isdir(temp_dir):
                shutil.rmtree(temp_dir)
            self.make_directory(temp_dir)
            if parallel:
                my_list = []
                for t_ in range(tstart_idx, tstop_idx, self.dt_video):
                    my_list.append((t_, t_df, temp_dir, t_start_row, t_stop_row))

                with multiprocessing.Pool(num_proc) as pool:
                    pool.starmap(self.make_frame, my_list)

            # else: # series 
            for t_ in range(t_start_row["t_idx"], t_stop_row["t_idx"], self.dt_video):
                if t_%10==0:
                    print(t_)
                self.make_frame(t_, t_df, temp_dir, t_start_row, t_stop_row)
                plt.close()

        if self.domain_size=="micro":
            figsize = (10,8)
        else:
            figsize = (16,9)
        self.matplotlib_writer(t_start_row["t_idx"], t_stop_row["t_idx"], temp_dir, figsize)

    def make_frame(self, t_, t_df, temp_dir, start_row, stop_row):
        fn = os.path.join(temp_dir, "f{}.png" .format(t_))

        # if self.domain_size == "estero":
        #     self.plot_timestep(t_hr=t_hr, fname=fn, t_start=t_start_xbeach, t_stop=t_stop_xbeach)
        if self.domain_size == "micro":
            self.plot_timestep_micro(t_=t_, t_df=t_df, fname=fn)
        plt.close()

    def plot_frame(self, t_hr):
        t_df = self.construct_time_df(self.hotstart_runs)
        t_plot_row = t_df.loc[(t_df["t_hr"] - t_hr).abs().idxmin()]
        
        print("creating frame at t_hr = {:.2f} hr" .format(t_hr))
        print("  nearest time step to input <{:.2f}> hr is t_hr = {:.2f} hr" .format(t_hr, t_plot_row["t_hr"]))
        print("  this corresponds to time array index: t_idx={}" .format(t_plot_row["t_idx"]))

        fn = "f{}.png" .format(t_plot_row["t_idx"])
        
        if self.domain_size == "micro":
            self.plot_timestep_micro(t_=t_plot_row["t_idx"], t_df=t_df, fname=fn)
        
    def plot_timestep_micro(self, t_, t_df, fname=None):
        t_row = t_df.loc[t_df["t_idx"]==t_]
        t_idx = t_row["t_idx_run"].item()
        hot_start_name = t_row["run"].item()
        model_dir = os.path.join(self.path_to_model, hot_start_name)
        xgr, ygr, _ = self.read_grid(model_dir)                                     # reading grid data
        data_plot = self.read_2d_data_xarray_timestep(var=self.var, t=t_idx, model_dir=model_dir)        # reading xbeach output


        # data_plot = data_plot - self.detrend_map
        mask = (data_plot < -99999)
        masked_array = np.ma.array(data_plot, mask=mask)

        # setting some info for the plot        
        cmap, cmap_bldg = self.setup_colormap()
        s, cbar_s = self.get_labels(t_row)

        # -- make figure and subplots
        figsize = (10,8)
        fig, (ax0, ax1) = plt.subplots(2,1, figsize=figsize, height_ratios=[8,1])

        # -- drawing first plot
        bldgs = self.read_buildings(model_dir)
        pcm = ax0.pcolormesh(xgr, ygr, masked_array, vmin=self.vmin, vmax=self.vmax, cmap=cmap)
        plt.colorbar(pcm, ax=ax0, extend="both", label=cbar_s, aspect=40)
        ax0.pcolormesh(xgr, ygr, bldgs, cmap=cmap_bldg)
        ax0.set_title(s)
        ax0.set_aspect("equal")

        self.draw_time_series(ax1, t_df, t_row["t_hr"].item())

        # --- saving figure
        self.save_fig(fig, 
                    fname, 
                    transparent=False, 
                    dpi=self.dpi,
                    )
    
    def setup_colormap(self):
        # setting up colormap for water
        if self.var == "H":
            cmap = mpl.cm.plasma
            cmap.set_bad('bisque')
        else:
            cmap = mpl.cm.plasma
            cmap.set_bad('bisque')

        # setting color for buildings.
        custom_color = 'springgreen'
        cmap_bldg = mpl.colors.ListedColormap([custom_color])
        cmap_bldg.set_bad(alpha=0)

        return cmap, cmap_bldg

    def construct_time_df(self, hotstart_runs):
        fn = os.path.join(self.path_to_model, hotstart_runs[0], "params.txt")
        dt = self.read_from_params(fn, "tintg")
        dt_run = self.read_from_params(fn, "tstop")
        t_tot = len(hotstart_runs)*dt_run
        t = np.arange(start=0, stop=t_tot, step=dt)
        t_idx = np.arange(start=0, stop=len(t), step=1)

        
        runs = []
        t_idx_run = []
        for hotstart in hotstart_runs:
            runs.extend([hotstart]*int(dt_run/dt))
            t_idx_run.extend(np.arange(start=0, stop=int(dt_run/dt), step=1))

        df = pd.DataFrame()
        df["t_sec"] = t
        df["t_hr"] = t/3600
        df["t_idx"] = t_idx
        df["run"] = runs
        df["t_idx_run"] = t_idx_run
        return df


    def get_labels(self, t_row):
        if self.var == "H":
            s = "Time: {:2.1f}h ({:8.0f}s)" .format(t_row["t_hr"].item(), t_row["t_sec"].item())
            cbar_s = "Wave Height (m)"
        elif self.var == "zs":
            s = "Time: {:2.1f}h ({:8.0f}s)" .format(t_row["t_hr"].item(), t_row["t_sec"].item())
            cbar_s = "Water Elevation (m)"
        elif self.var == "zs0":
            s = "Time {:2.1f}h ({:8.0f}s)" .format(t_row["t_hr"].item(), t_row["t_sec"].item())
            cbar_s = "Water Elevation - Tide Alone (m)"
        elif self.var == "zs1":
            s = "Time {:2.1f}h ({:8.0f}s)" .format(t_row["t_hr"].item(), t_row["t_sec"].item())
            cbar_s = "Water Elevation - Minus Tide (m)"
        return s, cbar_s


    def matplotlib_writer(self, tstart_idx, tstop_idx, temp_dir, figsize):
        video_name = '{}-{}.mp4'.format(self.model_runname, self.var)
        video_name = os.path.join(self.path_to_save_plot, video_name)
        fig, ax = plt.subplots(figsize=figsize)
        writer = animation.FFMpegWriter(fps=self.fps)
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

    def draw_time_series(self, ax, t_df, t_hr):
        # # -- now plotting time series
        forcing = self.frcing_to_dataframe()
        forcing["idx"] = range(0, len(forcing))
        x,y = forcing["t_hr"], forcing["el"]
        ax.plot(x,y, lw=0.75, ls="-.", color='k', zorder=0, label="ADCIRC/SWAN")
        
        s,e = self.xbeach_duration_to_start_stop(self.xbeach_duration)
        start = t_df.iloc[0]["t_hr"].item() + s
        stop  = t_df.iloc[-1]["t_hr"].item() + s

        start_idx = int(forcing.loc[(forcing["t_hr"] - start).abs().idxmin()]["idx"].item())
        stop_idx  = int(forcing.loc[(forcing["t_hr"] - stop).abs().idxmin()]["idx"].item())
        
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


    def set_hotstart_runs(self):
        hotstart_runs = [i for i in os.listdir(self.path_to_model) if os.path.isdir(os.path.join(self.path_to_model, i))]
        self.hotstart_runs = sorted(hotstart_runs)











