import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

from helpers.helpers import HelperFuncs

class PlotGrid(HelperFuncs):
    """docstring for plot_grid"""
    def __init__(self):
        super().__init__()

        # self.file_dir = os.path.dirname(os.path.realpath(__file__))
        # self.path_to_model = path_to_model
        
        # self.xgr = self.read_grd("x")
        # self.ygr = self.read_grd("y")
        # self.zgr = self.read_grd("z")

    def plot_dep_across_y(self, x_trans, lookonshore=False, drawdomain=False, savefig=False):
        xgr, ygr, zgr = self.read_grid()

        # --- first plotting domain
        if drawdomain:
            figd, axd = plt.subplots(1,1, figsize=(8,6))
            cmap = mpl.cm.BrBG_r
            cmap.set_bad('bisque',1.)
            axd.pcolormesh(xgr, ygr, zgr, vmin=-8.5, vmax=8.5, cmap=cmap)
        # ---

        fig, ax = plt.subplots(1,1, figsize=(8,6))
        colors = sns.color_palette("viridis", len(x_trans))
        cnt = 0
        for x_trans_ in x_trans:
            idx, _ = self.xy_to_grid_index(xgr, ygr, (x_trans_, 0))

            y_data = ygr[:,idx]
            z_data = zgr[:,idx]
            if lookonshore:
                ax.xaxis.set_inverted(True)  # inverted axis with autoscaling
                
            ax.plot(y_data, z_data, color=colors[cnt], label="x={}" .format(x_trans_))
            if drawdomain:
                x = xgr[0,idx]
                axd.axvline(x=x, ymin=0, ymax=np.shape(ygr)[1], color=colors[cnt], lw=2)
            cnt += 1
        
        ax.set_xlabel("y")
        ax.set_ylabel("z")
        ax.grid()
        ax.set_title("Elevation\nFrom Sea, Looking Towards Land")
        ax.set_ylim([-3.5, 1.])
        ax.legend(loc="upper right", ncols=len(x_trans))
        if savefig:
            self.save_fig(fig, "z.png",
                        transparent=False, 
                        dpi=300,
                        bbox_inches='tight',
                        pad_inches=0.1)

            if drawdomain:
                self.save_fig(figd, "domain.png",
                        transparent=False, 
                        dpi=300,
                        bbox_inches='tight',
                        pad_inches=0.1
                        )


if __name__ == "__main__":
    file_dir = os.path.dirname(os.path.realpath(__file__))              # current file directory
    model_runname = "test"
    path_to_model = os.path.join(file_dir, "..", "..", "xbeach", "models", model_runname)

    pg = PlotGrid(path_to_model=path_to_model)
    pg.plot_dep_across_y(x_trans=[0, 50, 100, 200, 300], 
                        lookonshore=True, 
                        drawdomain=False, 
                        savefig=False)
    
    plt.show()