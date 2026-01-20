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
        failed_bldgs = np.zeros(np.shape(elevated_bldgs), dtype=bool)
        # failed_bldgs[elevated_bldgs!=0] = 1        

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
                    failed_bldgs[bldg_] = True

        fn_out = os.path.join(self.path_to_save_plot, "removed_bldgs_elevated")
        np.save(fn_out, failed_bldgs)
