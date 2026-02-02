import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from helpers.helpers import HelperFuncs

class PlotCurrentQuiver(HelperFuncs):
    """docstring for plot_wave_heights"""
    def __init__(self):
        super().__init__()

    def plot(self, stat="velocity", domain_size="micro", fname=None):
        vmag = self.read_npy("{}_magnitude" .format(stat))
        vdir = self.read_npy("{}_direction" .format(stat))
        
        if stat == "velocity":
            scale = 30
            label = "1 m/s"
            U_key = 1
        elif stat == "bed_shear":
            scale = 300
            label = "10 $N/m^2$"
            U_key = 10

        ue = vmag*np.cos(np.deg2rad(vdir))
        ve = vmag*np.sin(np.deg2rad(vdir))

        xgr, ygr, zgr = self.read_grid()
        skip = (slice(None, None, 20), slice(None, None, 20))

        figsize = self.get_figsize(domain_size)
        figsize=(10,8)
        fig, ax = plt.subplots(1,1, figsize=figsize)
        

        q = ax.quiver(xgr[skip], ygr[skip], ue[skip], ve[skip], vmag[skip], scale=scale, width=0.004, cmap="viridis")
        ax.quiverkey(q, X=0.9, Y=1.02, U=U_key, label=label, labelpos='E')

        ax.pcolormesh(xgr, ygr, zgr, vmin=-8.5, vmax=8.5, cmap="BrBG_r", zorder=0)
        ax.set_aspect("equal")
        self.save_fig(fig, fname, transparent=True, dpi=1000)
