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
        self.threshold = 20 # newtons per m^2
        # self.safety_factor = 1.0

    def process(self, ):
        hsruns = self.set_hotstart_runs()
        elevated_bldgs = np.loadtxt(os.path.join(self.path_to_model, hsruns[0], "elevated_bldgs.grd"))
        failed_bldgs = np.zeros(np.shape(elevated_bldgs), dtype=bool)

        labeled_mask, num_features = ndi.label(elevated_bldgs)
        uplift_force = np.loadtxt(os.path.join(self.path_to_model, hsruns[-1], "stat_cumulative_uplift_impulse.dat"))
        for i in range(1, num_features):
            bldg_ = labeled_mask==i
            uplift_force_ = uplift_force[bldg_]
            # downward_force = self.down_pressure*floor_area
            if uplift_force_[0]>self.threshold:
                failed_bldgs[bldg_] = True

        fn_out = os.path.join(self.path_to_save_plot, "removed_bldgs_elevated")
        np.save(fn_out, failed_bldgs)
        