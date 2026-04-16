import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import os

from helpers.helpers import HelperFuncs

class ExtractStatsPoint(HelperFuncs):
    """docstring for xb_plotting_pt"""
    def __init__(self):
        super().__init__()

    def extract(self, 
            var, 
            xys, 
            pt_names,
            save_depth=False,
            t_start=None, 
            t_stop=None,
            drawdomain=False, 
            domain_size="estero", 
            moving_avg=False,
            window_sec=120,
            new_sec_step=120,
            savefig=False):

        xgr, ygr, zgr = self.read_grid()
        t = self.read_time_xarray()
        df = pd.DataFrame()
        df["t"] = t
        df.set_index("t", inplace=True)
        cnt = 0
        for xy in xys:
            idx, idy = self.xy_to_grid_index(xgr, ygr, xy)
            if var == "current":
                try:
                    ue = self.read_pt_data_xarray("uu", idx, idy)
                    ve = self.read_pt_data_xarray("vv", idx, idy)
                except:
                    ue = self.read_pt_data_xarray("ue", idx, idy)
                    ve = self.read_pt_data_xarray("ve", idx, idy)
                data_ = self.compute_velocity_mag(ue, ve, return_max=False)
            else:
                data_ = self.read_pt_data_xarray(var, idx, idy)
                if save_depth:
                    data_ = data_ - zgr[idy, idx]
                    data_[data_<0] = 0
                    data_ = np.nan_to_num(data_)

            df[pt_names[cnt]] = data_
    

            cnt += 1

        if moving_avg:
            df_new = pd.DataFrame()
            for col in pt_names:
                t, df_new[col] = self.calculate_running_avg(df.index, df[col].values, window_sec, new_sec_step)
            df_new["t"] = t            
            df_new.set_index("t", inplace=True)
            df = df_new.copy()


        if t_start == None:
            t_start = t[0]
        if t_stop == None:
            t_stop = t[-1]
        df = df.loc[t_start:t_stop]

        if save_depth:
            var = "water-depth"

        fn_out = os.path.join(self.path_to_save_plot, "{}-timeseries.csv" .format(var))
        df.to_csv(fn_out)

        if drawdomain:
            figsize = self.get_figsize(domain_size)
            fig, ax = plt.subplots(1,1, figsize=figsize)
            
            cmap = mpl.cm.BrBG_r
            cmap.set_bad('bisque',1.)
            ax.pcolormesh(xgr, ygr, zgr, vmin=-8.5, vmax=8.5, cmap=cmap)
            ax.set_xlabel("x (m)")
            ax.set_ylabel("y (m)")
            ax.set_aspect("equal")

            cnt = 0
            box_style = dict(
                boxstyle='round,pad=0.3', 
                facecolor='white', 
                edgecolor='k', 
                lw=0.1,
                alpha=0.9
                )
            for xy in xys:
                if "p" in pt_names[cnt]:
                    x_offset = 0
                    y_offset = -40
                else:
                    x_offset = 10
                    y_offset = 10
                x,y = xy[0], xy[1]
                ax.scatter(x, y, color='tomato',s=20, zorder=1)
                ax.annotate("{}" .format(pt_names[cnt]), (x+x_offset, y+y_offset), fontsize=7, bbox=box_style, zorder=2)
                cnt += 1

            if savefig:
                fn = "obs-points.png"
                self.save_fig(fig, fn, transparent=True, dpi=300)



