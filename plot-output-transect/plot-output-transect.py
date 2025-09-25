import os
import shutil
import numpy as np
import xarray as xr
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import seaborn as sns

class plot_transect():
    """docstring for xb_plotting_pt"""
    def __init__(self, model_runname, var="H"):
        self.file_dir = os.path.dirname(os.path.realpath(__file__))
        self.model_runname = model_runname
        self.path_to_model = os.path.join(self.file_dir, "..", "..", "xbeach", "models", self.model_runname)
        self.var = var
        self.xgr, self.ygr, self.zgr = self.read_grid()


    def var2label(self, var):
        if var == "H":
            ylabel = "Wave Height (m)"
        elif var == "zs":
            ylabel = "Water Level (m)"
        elif var == "zs0":
            ylabel = "zs0: Water Level - Surge/Tide Alone (m)"
        elif var=="zs1":
            ylabel = "zs1: Water Level - No Surge/Tide"
        return ylabel
    
    def read_data_xarray_transect(self, var, idy, t_idx, prnt_read=False, rtn_time_array=False):
        fn = os.path.join(self.path_to_model, "xboutput.nc")
        ds = xr.open_dataset(fn, chunks={"globaltime": 100})
        if prnt_read:
            print("Dataset object read:")
            print(ds)
            print("\n\n")
        time = ds["globaltime"].values
        # t_sec = t*3600
        # t_idx = np.argmin(np.abs(time - t_sec))
        slice_data = ds[var][t_idx,idy,:]

        if rtn_time_array:
            return slice_data.values, time

        return slice_data.values

    def read_data_xarray_gd(self, var="zb"):
        fn = os.path.join(self.path_to_model, "xboutput.nc")
        ds = xr.open_dataset(fn, chunks={"globaltime": 100})
        slice_data = ds[var][0,:,:]
        return slice_data



    def read_grid(self):
        fn = os.path.join(self.path_to_model, "x.grd")
        if os.path.isfile(fn):
            xgrid = os.path.join(self.path_to_model, "x.grd")
            ygrid = os.path.join(self.path_to_model, "y.grd")
            zgrid = os.path.join(self.path_to_model, "z.grd")

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
            fn_params = os.path.join(self.path_to_model, "params.txt")
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


    def plot_water_level_transect(self, y_trans, ts, plot_ground=True, 
                    h_plus_zs=False, fulldomain=True, drawdomain=False, 
                    dpi=300, legend=True, figsize=(10,4), fname=None):
        idy = np.argmin(np.abs(self.ygr[:,0] - y_trans))
    
        fig, ax = plt.subplots(1,1,figsize=figsize)
        colors = sns.color_palette("viridis")
        if plot_ground == True:
            grnd = self.zgr[idy,:]
            # _, time = self.read_data_xarray_transect(var="zb", idy=idy, t_idx=0, rtn_time_array=True)
        fn = os.path.join(self.path_to_model, "xboutput.nc")
        ds = xr.open_dataset(fn, chunks={"globaltime": 100})
        time = ds["globaltime"].values

        # get data for variable
        for t_i, t in enumerate(ts):
            t_sec = t*3600
            t_idx = np.argmin(np.abs(time - t_sec))

            print("  found nearest time step as to t = {:.2f} hr is t = {:.2f} hr" .format(t, time[t_idx]/3600))

            data_ = self.read_data_xarray_transect(var=self.var, idy=idy, t_idx=t_idx)
            data_[data_<-99999] = 0
            c = colors[t_i]

            if h_plus_zs:
                data_zs = self.read_data_xarray_transect(var="zs", idy=idy, t_idx=t_idx)
                data_zs1 = self.read_data_xarray_transect(var="zs1", idy=idy, t_idx=t_idx)
                data_zs0 = self.read_data_xarray_transect(var="zs0", idy=idy, t_idx=t_idx)

                data_zs[data_<-99999] = 0
                # data_tot = data_zs + data_

                # ax.plot(data_tot, color=c, lw=2, label="H+zs" .format(t))

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
        if fname!=None:
            # fn = "ytrans{}-t{}.png" .format(y_trans, t)
            plt.savefig(fname,
                        transparent=False, 
                        dpi=300,
                        bbox_inches='tight',
                        pad_inches=0.1,
                        )
            plt.close()


        if drawdomain:
            data_plot = self.read_data_xarray_gd()


            if fulldomain:
                figsize=(4,8)
            else:
                figsize=(8,6)


            fig, ax = plt.subplots(1,1, figsize=figsize)
            # --- new
            mask = (data_plot < -99999)
            masked_array = np.ma.array(data_plot, mask=mask)
            cmap = mpl.cm.BrBG_r
            cmap.set_bad('bisque',1.)
            ax.pcolormesh(self.xgr, self.ygr, masked_array, vmin=-8.5, vmax=8.5, cmap=cmap)
            cnt = 0

            y = self.ygr[idy,0]
            ax.axhline(y=y, xmin=0, xmax=np.shape(data_plot)[1], color='k', lw=2)


            if fname!=None:
                fn = "{}-trns.png" .format(fname)
                plt.savefig(fn,
                            transparent=False, 
                            dpi=300,
                            bbox_inches='tight',
                            pad_inches=0.1,
                            )



    def video_transect(self, y_trans, t_start=None, t_stop=None, h_plus_zs=True, dpi=300):
        idy = np.argmin(np.abs(self.ygr[:,0] - y_trans))

        grnd, time = self.read_data_xarray_transect(var="zb", idy=idy, t_idx=0, rtn_time_array=True)
        if t_start == None:
            tstart = time[0]
        if t_stop == None:
            t_stop = time[-1]/3600


        tstart_idx = np.argmin(np.abs(time-t_start*3600))
        tstop_idx = np.argmin(np.abs(time-t_stop*3600))
        temp_dir = os.path.join(self.file_dir, "temp")
        figsize=(10,5)
        for t_idx in range(tstart_idx, tstop_idx):
            self.make_directory(temp_dir)

            fn = os.path.join(temp_dir, "f{}.png" .format(t_idx))
            self.plot_water_level_transect(
                                    y_trans=y_trans, 
                                    ts=[time[t_idx]/3600],
                                    h_plus_zs=h_plus_zs, 
                                    legend=False,
                                    figsize=figsize,
                                    dpi=dpi,
                                    fname=fn,
                                    )


        self.matplotlib_writer(tstart_idx=tstart_idx, 
                               tstop_idx=tstop_idx, 
                               temp_dir=temp_dir, 
                               figsize=figsize, 
                               dpi=dpi)


    def make_directory(self, path_out):
        if not os.path.exists(path_out):
            os.makedirs(path_out)
        return path_out



    def matplotlib_writer(self, tstart_idx, tstop_idx, temp_dir, figsize, dpi):
        video_name = '{}-{}.mp4'.format(self.model_runname, self.var)
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
        # if os.path.isdir(temp_dir):
        #     shutil.rmtree(temp_dir)

if __name__ == "__main__":

    pt = plot_transect("test", var="H")
    pt.plot_water_level_transect(y_trans=400,
                                 ts=[1],
                                 h_plus_zs=False,
                                 drawdomain=False, 
                                 fulldomain=False,
                                 # fname="run26_t1hr.png"
                                 )

    # pt.video_transect(y_trans=400,
    #                   t_start=1,
    #                   t_stop=1.5,
    #                   h_plus_zs=True,
    #                   dpi=100,
    #                   )

    plt.show()

