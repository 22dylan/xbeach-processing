import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

from helpers.helpers import HelperFuncs

class PlotOutputPoint(HelperFuncs):
    """docstring for xb_plotting_pt"""
    def __init__(self):
        super().__init__()

    def plot(self, var, xys, drawdomain=False, fulldomain=True, savefig=False):
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
            ax.plot(t/3600, data_, label="{}" .format(cnt), color=colors[cnt], lw=1.3)
 
            cnt += 1

        ax.set_xlabel("Time (hrs)")
        s, _, _ = self.var2label(var)
        ax.set_ylabel(s)
        ax.legend()
        if savefig:
            fn = "elevation-timeseries.png"
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









