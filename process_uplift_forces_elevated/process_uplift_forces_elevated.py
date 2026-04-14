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

    def process(self):
        ffe_data = pd.read_csv(self.path_to_dmg)
        ffe_data.set_index("VDA_id", inplace=True)

        fn = os.path.join(self.path_to_save_plot, "stats_at_bldgs.csv")
        bldg_df = pd.read_csv(fn)
        bldg_df.set_index("VDA_id", inplace=True)

        elevated_bldgs, _ = self.get_elevated_bldgs(ffe_data)
        ffe_data = ffe_data.loc[elevated_bldgs.index]
        ffe_data["TA_ShapeSTArea_Sqm"] = ffe_data["TA_ShapeSTArea_Sqft"]*0.3048*0.3048
        bldg_df = pd.merge(bldg_df, ffe_data, left_index=True, right_index=True)

        # -- new
        bldg_df["freeboard_surge_wave"] = bldg_df["max_stat_zs"] - (bldg_df["FFE_ffe_ft"]*0.3048)
        threshold = []
        remove_bldg = []
        for bldg_i, bldg in bldg_df.iterrows():
            if bldg["TA_ActYearBuilt"]>=1974:
                remove_bldg.append(False)
                continue

            else:
                if bldg["freeboard_surge_wave"]<2.25:
                    remove_bldg.append(False)
                else:
                    remove_bldg.append(True)

        bldg_df["remove_elevated"] = remove_bldg
        bldg_df["elevated"] = True
        bldg_df = bldg_df[["elevated", "remove_elevated"]]

        fn_out = os.path.join(self.path_to_save_plot, "removed_elevated_bldgs.csv")
        bldg_df.to_csv(fn_out)
        

