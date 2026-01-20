import os
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib as mpl
import scipy.ndimage as ndi

from helpers.helpers import HelperFuncs


class ProcessUpliftForcesElevated(HelperFuncs):
    """docstring for plot_wave_heights"""
    def __init__(self):
        super().__init__()
        self.rho = 1024
        self.g = 9.81
        self.down_pressure = 5000 # newtons per m^2
        self.safety_factor = 1.0

    def process(self, ):
        hsruns = self.set_hotstart_runs()
        elevated_bldgs = np.loadtxt(os.path.join(self.path_to_model, hsruns[0], "elevated_bldgs.grd"))
        failed_bldgs = np.zeros(np.shape(elevated_bldgs))
        failed_bldgs[elevated_bldgs!=0] = 1        

        labeled_mask, num_features = ndi.label(elevated_bldgs)
        dx, dy = self.get_resolution(os.path.join(self.path_to_model, hsruns[0]))
        res = dx*dy

        struct = ndi.generate_binary_structure(2, 2)
        for hs in hsruns:
            e = np.loadtxt(os.path.join(self.path_to_model, hs, "water_elev_out.dat"))
            z = np.loadtxt(os.path.join(self.path_to_model, hs, "z.grd"))
            depth = e-z
            
            for i in range(1, num_features):
                bldg_ = labeled_mask==i
                floor_area = np.sum(bldg_)*res  # floor area (m^2)
            
                p = ndi.binary_dilation(bldg_, structure=struct)
                p = p & ~bldg_
                d_perimeter = depth[p]
                ffe = elevated_bldgs[bldg_][0]
                free_board = d_perimeter - ffe
                free_board[free_board<0] = 0
                uplift_force = self.rho*self.g*np.average(free_board)*floor_area
                downward_force = self.down_pressure*floor_area
                if uplift_force>(downward_force*self.safety_factor):
                    failed_bldgs[bldg_] = 2

        fn_out = os.path.join(self.path_to_save_plot, "elevated_bldgs_failed")
        np.save(fn_out, failed_bldgs)
        
    def plot(self, grey_background=True, domain_size="estero"):
        fn_failed = os.path.join(self.path_to_save_plot, "elevated_bldgs_failed.npy")
        failed_bldgs = np.load(fn_failed)
        mask = (failed_bldgs == 0)
        failed_bldgs = np.ma.array(failed_bldgs, mask=mask)
        
        cmap = mpl.colors.ListedColormap(["darkseagreen", "red"])
        if grey_background:
            cmap.set_bad("grey")
        else:
            cmap.set_bad(alpha=0.5)
        fn_params = os.path.join(self.path_to_model, "params.txt")
        if os.path.exists(fn_params):
            model_dir = self.path_to_model
        else:
            hs0 = self.set_hotstart_runs()[0]
            model_dir = os.path.join(self.path_to_model, hs0)
            fn_params = os.path.join(model_dir, "params.txt")

        xgr, ygr, zgr = self.read_grid(model_dir)
        figsize = self.get_figsize(domain_size)
        fig, ax = plt.subplots(1,1, figsize=figsize)
        ax.pcolormesh(xgr, ygr, zgr, vmin=-8.5, vmax=8.5, cmap="BrBG_r", zorder=0)
        pcm = ax.pcolormesh(xgr, ygr, failed_bldgs, cmap=cmap, zorder=1)
        # ax.imshow(H_plot, cmap=cmap, origin="lower")

        # plt.colorbar(pcm, ax=ax, extend="max", label=labl, aspect=40)
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.set_aspect("equal")

        plt.show()