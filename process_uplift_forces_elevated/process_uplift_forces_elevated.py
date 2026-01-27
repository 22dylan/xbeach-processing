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
        # self.safety_factor = 1.0

    def process(self, threshold=20):
        """
        threshold: threshold to remove buildings (units: kN-hr)
        """
        hsruns = self.set_hotstart_runs()
        model_dir = self.get_first_model_dir()
        fn = os.path.join(model_dir, "elevated_bldgs.grd")
        elevated_bldgs = np.loadtxt(fn)
        failed_bldgs = np.zeros(np.shape(elevated_bldgs), dtype=bool)

        labeled_mask, num_features = ndi.label(elevated_bldgs)
        uplift_force = np.loadtxt(os.path.join(self.path_to_model, hsruns[-1], "stat_cumulative_uplift_impulse.dat"))
        for i in range(1, num_features):
            bldg_ = labeled_mask==i
            uplift_force_ = uplift_force[bldg_]
            # downward_force = self.down_pressure*floor_area
            if uplift_force_[0]>threshold:
                failed_bldgs[bldg_] = True

        fn_out = os.path.join(self.path_to_save_plot, "removed_bldgs_elevated.dat")
        np.savetxt(fn_out, failed_bldgs)
        