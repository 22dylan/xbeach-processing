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

    # def process(self, threshold=20):
    #     """
    #     threshold: threshold to remove buildings (units: kN-hr)
    #     """
    #     hsruns = self.set_hotstart_runs()
    #     model_dir = self.get_first_model_dir()
    #     fn = os.path.join(model_dir, "elevated_bldgs.grd")
    #     elevated_bldgs = np.loadtxt(fn)
    #     failed_bldgs = np.zeros(np.shape(elevated_bldgs), dtype=bool)

    #     labeled_mask, num_features = ndi.label(elevated_bldgs)
    #     dx, dy = self.get_resolution()
    #     uplift_force = np.loadtxt(os.path.join(self.path_to_model, hsruns[-1], "stat_cumulative_uplift_impulse.dat"))
    #     for i in range(1, num_features):
    #         bldg_ = labeled_mask==i
    #         floor_area = np.sum(bldg_)*dx*dy
    #         downward_force_ = threshold*floor_area
    #         uplift_force_ = uplift_force[bldg_]
    #         # downward_force = self.down_pressure*floor_area
    #         if uplift_force_[0]>downward_force_:
    #             failed_bldgs[bldg_] = True
        
    #     fn_out = os.path.join(self.path_to_save_plot, "removed_bldgs_elevated.dat")
    #     np.savetxt(fn_out, failed_bldgs)
    
    def process2(self, bldg_df):
        ffe_data = pd.read_csv(self.path_to_dmg)
        ffe_data.set_index("VDA_id", inplace=True)
        bldg_df.set_index("VDA_id", inplace=True)

        elevated_bldgs, _ = self.get_elevated_bldgs(ffe_data)
        ffe_data = ffe_data.loc[elevated_bldgs.index]
        ffe_data["TA_ShapeSTArea_Sqm"] = ffe_data["TA_ShapeSTArea_Sqft"]*0.3048*0.3048
        bldg_df = pd.merge(bldg_df, ffe_data, left_index=True, right_index=True)
        
        threshold = []
        for bldg_i, bldg in bldg_df.iterrows():
            if bldg["TA_ActYearBuilt"] > 1974:
                threshold.append(5*bldg["TA_ShapeSTArea_Sqm"])
                continue
            elif bldg["VDA_breakaway_walls"] == "yes":
                threshold.append(0.75*bldg["TA_ShapeSTArea_Sqm"])
                continue
            else:
                threshold.append(0.5*bldg["TA_ShapeSTArea_Sqm"])


        bldg_df["uplift_threshold"] = threshold
        bldg_df["remove_elevated"] = bldg_df["uplift_impulse"]>bldg_df["uplift_threshold"]
        bldg_df = bldg_df[["elevated", "uplift_impulse", "uplift_threshold", "remove_elevated", "FFE_ffe_ft", "TA_ActYearBuilt", "VDA_breakaway_walls", "TA_ShapeSTArea_Sqm"]]
        fn_out = os.path.join(self.path_to_save_plot, "removed_elevated_bldgs.csv")
        bldg_df.to_csv(fn_out)

        # fn_out = os.path.join(self.path_to_save_plot, "temp2.csv")
        # bldg_df.to_csv(fn_out)

