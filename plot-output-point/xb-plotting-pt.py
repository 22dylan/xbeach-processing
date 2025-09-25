import os
import numpy as np
import xarray as xr
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.signal as sg

class xb_plotting_pt():
    """docstring for xb_plotting_pt"""
    def __init__(self, model_runname, var="H"):
        self.file_dir = os.path.dirname(os.path.realpath(__file__))
        self.model_runname = model_runname
        self.path_to_model = os.path.join(self.file_dir, "..", "..", "xbeach", "models", self.model_runname)
        self.var = var
        self.xgr, self.ygr = self.read_grid()
    
    def read_data_xarray_pt(self, var, idx, idy, prnt_read=False, rtn_time_array=False):
        fn = os.path.join(self.path_to_model, "xboutput.nc")
        ds = xr.open_dataset(fn, chunks={"globaltime": 100})
        if prnt_read:
            print("Dataset object read:")
            print(ds)
            print("\n\n")
        
        slice_data = ds[var][:,idy,idx]

        if rtn_time_array:
            time = ds["globaltime"].values
            return slice_data.values, time
        else:
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

        return xgr, ygr

    def compute_Hs(self, H):
        H_one_third = np.quantile(H, q=2/3)
        H = H[H>H_one_third]
        Hs = np.mean(H)
        return Hs

    def plot_water_level_point(self, xys, drawdomain=False, fulldomain=True, savefig=False):
        colors = sns.color_palette("husl")
        fig, ax = plt.subplots(1,1,figsize=(6,4))
        cnt = 0

        for xy in xys:
            idx = np.argmin(np.abs(self.xgr[0,:] - xy[0]))
            idy = np.argmin(np.abs(self.ygr[:,0] - xy[1]))
            data_, t= self.read_data_xarray_pt(var=self.var, idx=idx, idy=idy, prnt_read=False, rtn_time_array=True)
            print(np.mean(data_))

            peaks,_ = sg.find_peaks(data_)
            trghs,_ = sg.find_peaks(np.negative(data_))
            n_obs = np.minimum(len(peaks), len(trghs))
            peaks = peaks[0:n_obs]
            trghs = trghs[0:n_obs]

            H = data_[peaks] - data_[trghs]
            Hs = self.compute_Hs(H)
            print("Hs at {}: {}" .format(xy, Hs))

            # peaks, troughs = np.array(peaks), np.array(troughs)

            # -- plotting
            
            ax.plot(t/3600, data_, label="{}" .format(cnt), color=colors[cnt], lw=1.3)
 
            cnt += 1

        ax.set_xlabel("Time (hrs)")
        # ax.set_ylabel("Elevation (m)")
        ax.set_ylabel("Wave Height (m)")
        ax.legend()
        if savefig:
            fn = "elevation-timeseries.png"
            plt.savefig(fn,
                        transparent=False, 
                        dpi=300,
                        bbox_inches='tight',
                        pad_inches=0.1,
                        )

        if drawdomain:
            data_plot = self.read_data_xarray_gd()
            # data_plot = self.data[0,:,:]

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
            ax.set_xlabel("x (m)")
            ax.set_ylabel("y (m)")

            cnt = 0

            for xy in xys:
                idx = np.argmin(np.abs(self.xgr[0,:] - xy[0]))
                idy = np.argmin(np.abs(self.ygr[:,0] - xy[1]))
                # -- new
                x, y = self.xgr[0,idx], self.ygr[idy, 0]                

                # -- old
                # x, y = self.xgr[0,xy[0]], self.ygr[xy[1],0]
                ax.scatter(x, y, color=colors[cnt],s=50)
                ax.annotate("{}" .format(cnt), (x, y))

                # ax.scatter(xy[0], xy[1], color=colors[cnt],s=50)
                # ax.annotate("{}" .format(cnt), (xy[0], xy[1]))
                cnt += 1

            if savefig:
                fn = "obs-points.png"
                plt.savefig(fn,
                            transparent=False, 
                            dpi=300,
                            bbox_inches='tight',
                            pad_inches=0.1,
                            )


if __name__ == "__main__":
    xbpp = xb_plotting_pt("run26", var="zs1")
    xbpp.plot_water_level_point(xys=[
                                    [4,0], 
                                    # [200,400], 
                                    # [360,400], 
                                    # [373,400], 
                                    # [600,400]
                                    ], 
                                drawdomain=True, 
                                fulldomain=False, 
                                savefig=False)


    plt.show()

