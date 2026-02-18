import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from helpers.helpers import HelperFuncs

class PlotOutputPoint(HelperFuncs):
    """docstring for xb_plotting_pt"""
    def __init__(self):
        super().__init__()

    def plot_timeseries(self, 
            var, 
            xys, 
            t_start=None, 
            t_stop=None, 
            x_units="hr",
            drawdomain=False, 
            fulldomain=True, 
            savefig=False):
        xgr, ygr, zgr = self.read_grid()

        colors = sns.color_palette("husl")
        fig0, ax = plt.subplots(1,1,figsize=(6,4))
        cnt = 0
        t = self.read_time_xarray()
        for xy in xys:
            idx, idy = self.xy_to_grid_index(xgr, ygr, xy)
            data_ = self.read_pt_data_xarray(var, idx, idy)

            H = self.get_H(data_)
            Hs = self.compute_Hs(H)
            print("Hs at {}: {}" .format(xy, Hs))

            # -- plotting
            if x_units == "hr":
                ax.plot(t/3600, data_, label="{}" .format(cnt), color=colors[cnt], lw=1.3)
                xlbl = "Time (hrs)"
            elif x_units == "sec":
                ax.plot(t, data_, label="{}" .format(cnt), color=colors[cnt], lw=1.3)
                xlbl = "Time (sec)"

 
            cnt += 1

        ax.set_xlabel(xlbl)
        s, _, _ = self.var2label(var)
        ax.set_ylabel(s)
        ax.legend()

        xlim = ax.get_xlim()
        if t_start == None:
            t_start = xlim[0]
        if t_stop == None:
            t_stop = xlim[1]
        xlim = (t_start, t_stop)
        ax.set_xlim(xlim)

        if savefig:
            fn = "{}-timeseries.png" .format(var)
            self.save_fig(fig0, fn, transparent=True, dpi=300)

        if drawdomain:
            if fulldomain:
                figsize=(4,8)
            else:
                figsize=(8,6)
            fig1, ax = plt.subplots(1,1, figsize=figsize)
            
            cmap = mpl.cm.BrBG_r
            cmap.set_bad('bisque',1.)
            ax.pcolormesh(xgr, ygr, zgr, vmin=-8.5, vmax=8.5, cmap=cmap)
            ax.set_xlabel("x (m)")
            ax.set_ylabel("y (m)")

            cnt = 0
            for xy in xys:
                x,y = xy[0], xy[1]
                ax.scatter(x, y, color=colors[cnt],s=50)
                ax.annotate("{}" .format(cnt), (x, y))
                cnt += 1

            if savefig:
                fn = "obs-points.png"
                self.save_fig(fig1, fn, transparent=True, dpi=300)

    def plot_Hs(self, var, xys, chunk_size_min=15, t_start=None, t_stop=None, drawdomain=False, fulldomain=True, savefig=False):
        chunk_size_sec = chunk_size_min*60
        xgr, ygr, zgr = self.read_grid()
        t = self.read_time_xarray()

        if len(t) % chunk_size_sec != 0:
            steps_to_trim = len(t) % chunk_size_sec
            t = t[steps_to_trim:]
        
        t_chunks = np.arange(t[0], t[-1], step=chunk_size_sec)
        t_idxs = [self.time_to_tindex(t_,t) for t_ in t_chunks] # get indicies to chunk out data
        
        colors = sns.color_palette("husl")
        fig0, ax = plt.subplots(1,1,figsize=(6,4))
        cnt = 0
        for xy in xys:
            idx, idy = self.xy_to_grid_index(xgr, ygr, xy)
            data_ = self.read_pt_data_xarray(var, idx, idy)
            data_ = data_[steps_to_trim:]

            Hs = []
            t_idx_prior = 0
            for t_idx in t_idxs[1:]:
                H = self.get_H(data_[t_idx_prior:t_idx])
                Hs_ = self.compute_Hs(H)
                Hs.append(Hs_)
                t_idx_prior = t_idx

            # -- plotting
            ax.plot(t_chunks[1:]/3600, Hs, label="{}" .format(cnt), color=colors[cnt], lw=1.3)
            cnt += 1

        ax.set_xlabel("Time (hrs)")
        s, _, _ = self.var2label(var)
        ax.set_ylabel("Significant Wave Height (m)")
        ax.legend()

        xlim = ax.get_xlim()
        if t_start == None:
            t_start = xlim[0]
        if t_stop == None:
            t_stop = xlim[1]
        xlim = (t_start, t_stop)
        ax.set_xlim(xlim)

        if savefig:
            fn = "Hs-timeseries.png"
            self.save_fig(fig0, fn, transparent=True, dpi=300)

        if drawdomain:
            if fulldomain:
                figsize=(4,8)
            else:
                figsize=(8,6)
            fig1, ax = plt.subplots(1,1, figsize=figsize)
            
            cmap = mpl.cm.BrBG_r
            cmap.set_bad('bisque',1.)
            ax.pcolormesh(xgr, ygr, zgr, vmin=-8.5, vmax=8.5, cmap=cmap)
            ax.set_xlabel("x (m)")
            ax.set_ylabel("y (m)")

            cnt = 0
            for xy in xys:
                x,y = xy[0], xy[1]
                ax.scatter(x, y, color=colors[cnt],s=50)
                ax.annotate("{}" .format(cnt), (x, y))
                cnt += 1
            ax.set_aspect("equal")
            
            if savefig:
                fn = "obs-points.png"
                self.save_fig(fig1, fn, transparent=True, dpi=300)








