import os
import shutil
import numpy as np
import xarray as xr
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import seaborn as sns

from helpers.helpers import HelperFuncs

class PlotOutputTransect(HelperFuncs):
    """docstring for xb_plotting_pt"""
    def __init__(self):
        super().__init__()

    def plot_water_level_transect(self, var, y_trans, ts, plot_ground=True, 
                    h_plus_zs=False, fulldomain=True, drawdomain=False, 
                    dpi=300, legend=True, figsize=(10,4), fname=None):
        xgr, ygr, zgr = self.read_grid()
        _, idy = self.xy_to_grid_index(xgr, ygr, (0, y_trans))

        fig0, ax = plt.subplots(1,1,figsize=figsize)
        colors = sns.color_palette("viridis")
        if plot_ground == True:
            grnd = zgr[idy,:]

        time = self.read_time_xarray()

        # get data for variable
        for t_i, t in enumerate(ts):
            t_sec = t*3600
            t_idx = np.argmin(np.abs(time - t_sec))
            print("  found nearest time step as to t = {:.3f} hr is t = {:.3f} hr" .format(t, time[t_idx]/3600))

            data_ = self.read_transect_data_xarray(var=var, idy=idy, t_idx=t_idx)
            data_[data_<-99999] = 0
            c = colors[t_i]

            if h_plus_zs:
                data_zs = self.read_transect_data_xarray(var="zs", idy=idy, t_idx=t_idx)
                data_zs1 = self.read_transect_data_xarray(var="zs1", idy=idy, t_idx=t_idx)
                data_zs0 = self.read_transect_data_xarray(var="zs0", idy=idy, t_idx=t_idx)

                data_zs[data_<-99999] = 0
                ax.plot(data_zs, color="grey", ls="-", lw=1, label="zs")
                ax.plot(data_zs0, color="red", ls="-.", lw=1, label="zs0")
                ax.plot(data_zs1, color="blue", ls="-.", lw=1, label="zs1")
                
                s_title = "water elevation at y={} m\nt={:.2f} hr ({:.0f} s)" .format(y_trans, time[t_idx]/3600, time[t_idx])
            else:
                ax.plot(data_, color=c, lw=2, label="{:.2f} hr" .format(t))
                s_title = "water elevation at y={} m" .format(y_trans)

        if plot_ground:
            ax.plot(grnd, 'k')

        ax.set_xlabel("x")
        # ax.set_ylabel(ylabel)
        ylim = ax.get_ylim()
        ax.set_ylim([ylim[0], 6])
        ax.set_xlim([0,np.shape(data_)[0]])
        ax.set_title(s_title)
        if legend:
            ax.legend()
        if fname != None:
            self.save_fig(fig0, fname, transparent=True, dpi=300)

        if drawdomain:

            if fulldomain:
                figsize=(4,8)
            else:
                figsize=(8,6)

            fig1, ax = plt.subplots(1,1, figsize=figsize)
            # --- new
            cmap = mpl.cm.BrBG_r
            cmap.set_bad('bisque',1.)
            ax.pcolormesh(xgr, ygr, zgr, vmin=-8.5, vmax=8.5, cmap=cmap)
            cnt = 0

            y = ygr[idy,0]
            ax.axhline(y=y, xmin=0, xmax=np.shape(zgr)[1], color='k', lw=2)

            if fname != None:
                fname = "domain-transect"
                self.save_fig(fig1, fname, transparent=True, dpi=300)

    def video_transect(self, var, y_trans, t_start=None, t_stop=None, h_plus_zs=True, dpi=300):
        xgr, ygr, zgr = self.read_grid()
        _, idy = self.xy_to_grid_index(xgr, ygr, (0, y_trans))

        time = self.read_time_xarray()
        grnd = zgr[idy,:]

        if t_start == None:
            tstart = time[0]
        if t_stop == None:
            t_stop = time[-1]/3600

        tstart_idx = np.argmin(np.abs(time-t_start*3600))
        tstop_idx = np.argmin(np.abs(time-t_stop*3600))
        temp_dir = os.path.join(self.file_dir, "temp")
        figsize=(10,5)
        self.make_directory(temp_dir)
        for t_idx in range(tstart_idx, tstop_idx):
            fn = os.path.join(temp_dir, "f{}.png" .format(t_idx))
            self.plot_water_level_transect(var=var,
                                    y_trans=y_trans, 
                                    ts=[time[t_idx]/3600],
                                    h_plus_zs=h_plus_zs, 
                                    legend=False,
                                    figsize=figsize,
                                    dpi=dpi,
                                    fname=fn,
                                    )


        self.matplotlib_writer(var=var,
                               tstart_idx=tstart_idx, 
                               tstop_idx=tstop_idx, 
                               temp_dir=temp_dir, 
                               figsize=figsize, 
                               dpi=dpi)


    def matplotlib_writer(self, var, tstart_idx, tstop_idx, temp_dir, figsize, dpi):
        video_name = '{}-{}-transect.mp4'.format(self.model_runname, var)
        video_name = os.path.join(self.path_to_save_plot, video_name)
        fig, ax = plt.subplots(figsize=figsize)
        writer = animation.FFMpegWriter(fps=10)
        with writer.saving(fig, video_name, dpi=300):
          for step in range(tstart_idx, tstop_idx):
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


