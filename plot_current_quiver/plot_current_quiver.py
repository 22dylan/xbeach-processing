import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from helpers.helpers import HelperFuncs

class PlotCurrentQuiver(HelperFuncs):
    """docstring for plot_wave_heights"""
    def __init__(self):
        super().__init__()

    def plot(self, domain_size="micro"):
        vmag = self.read_npy("velocity_magnitude")
        vdir = self.read_npy("velocity_direction")
        
        ue = vmag*np.cos(np.deg2rad(vdir))
        ve = vmag*np.sin(np.deg2rad(vdir))

        xgr, ygr, zgr = self.read_grid()
        skip = (slice(None, None, 4), slice(None, None, 4))

        figsize = self.get_figsize(domain_size)
        figsize=(10,8)
        fig, ax = plt.subplots(1,1, figsize=figsize)
        
        q = ax.quiver(xgr[skip], ygr[skip], ue[skip], ve[skip], vmag[skip], scale=20, width=0.004, cmap="viridis")
        ax.quiverkey(q, X=0.9, Y=1.02, U=1, label='1 m/s', labelpos='E')

        ax.pcolormesh(xgr, ygr, zgr, vmin=-8.5, vmax=8.5, cmap="BrBG_r", zorder=0)
        ax.set_aspect("equal")