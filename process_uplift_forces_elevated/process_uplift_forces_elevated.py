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

    def process(self, bldg_df):
        ffe_data = pd.read_csv(self.path_to_dmg)
        ffe_data.set_index("VDA_id", inplace=True)
        bldg_df.set_index("VDA_id", inplace=True)

        fn = os.path.join(self.path_to_save_plot, "stats_at_bldgs.csv")
        stats = pd.read_csv(fn)
        stats.set_index("VDA_id", inplace=True)
        bldg_df = pd.merge(bldg_df, stats, left_index=True, right_index=True)        

        elevated_bldgs, _ = self.get_elevated_bldgs(ffe_data)
        ffe_data = ffe_data.loc[elevated_bldgs.index]
        ffe_data["TA_ShapeSTArea_Sqm"] = ffe_data["TA_ShapeSTArea_Sqft"]*0.3048*0.3048
        bldg_df = pd.merge(bldg_df, ffe_data, left_index=True, right_index=True)
        


        # # -- old ------
        # threshold = []
        # for bldg_i, bldg in bldg_df.iterrows():
        #     if bldg["TA_ActYearBuilt"] > 1974:
        #         threshold.append(5*bldg["TA_ShapeSTArea_Sqm"])
        #         continue
        #     elif bldg["VDA_breakaway_walls"] == "yes":
        #         threshold.append(1*bldg["TA_ShapeSTArea_Sqm"])
        #         continue
        #     else:
        #         threshold.append(0.75*bldg["TA_ShapeSTArea_Sqm"]*bldg["TA_MaxStories"])

        # bldg_df["uplift_threshold"] = threshold
        # bldg_df["remove_elevated"] = bldg_df["uplift_impulse"]>bldg_df["uplift_threshold"]
        # bldg_df = bldg_df[["elevated", "uplift_impulse", "uplift_threshold", "remove_elevated", "FFE_ffe_ft", "TA_ActYearBuilt", "VDA_breakaway_walls", "TA_ShapeSTArea_Sqm"]]
        # # -------------

        # -- new
        bldg_df["freeboard_surge_wave"] = bldg_df["stat_max_zs"] - (bldg_df["FFE_ffe_ft"]*0.3048)
        threshold = []
        remove_bldg = []
        for bldg_i, bldg in bldg_df.iterrows():
            if bldg["TA_ActYearBuilt"]>=1974:
                remove_bldg.append(False)
                continue
                # if bldg["freeboard_surge_wave"]<99999999:
                #     remove_bldg.append(False)
                # else:
                #     remove_bldg.append(True)
                # continue

            else:
                if bldg["freeboard_surge_wave"]<2.25:
                    remove_bldg.append(False)
                else:
                    remove_bldg.append(True)

            # if bldg["freeboard_surge_wave"]<2.25:
            #     remove_bldg.append(False)
            #     continue
            # if bldg["TA_ActYearBuilt"]<1974:
            #     remove_bldg.append(True)

                # if bldg["freeboard_surge_wave"]<=2.6:
                #     remove_bldg.append(False)
                # else:
                #     remove_bldg.append(True)

        bldg_df["remove_elevated"] = remove_bldg
        bldg_df = bldg_df[["elevated", "uplift_impulse", "remove_elevated", "FFE_ffe_ft", "TA_ActYearBuilt", "VDA_breakaway_walls", "TA_ShapeSTArea_Sqm"]]

        # -- new ------
        # threshold = []
        # remove_bldg = []
        # for bldg_i, bldg in bldg_df.iterrows():
        #     if bldg["uplift_impulse"]<0.726:
        #         remove_bldg.append(False)       # standing
        #         continue
        #     if bldg["VDA_breakaway_walls"] == "yes":
        #         remove_bldg.append(False)       # standing
        #         continue
        #     else:
        #         if bldg["TA_ShapeSTArea_Sqft"]<1700:
        #             remove_bldg.append(True)
        #         else:
        #             remove_bldg.append(False)

        # bldg_df["remove_elevated"] = remove_bldg
        # bldg_df = bldg_df[["elevated", "uplift_impulse", "remove_elevated", "FFE_ffe_ft", "TA_ActYearBuilt", "VDA_breakaway_walls", "TA_ShapeSTArea_Sqm"]]
        # -------------

        
        

        fn_out = os.path.join(self.path_to_save_plot, "removed_elevated_bldgs.csv")
        bldg_df.to_csv(fn_out)
        




