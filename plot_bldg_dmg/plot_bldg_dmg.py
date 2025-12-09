import os
import numpy as np
import pandas as pd
import geopandas as gpd

import matplotlib.pyplot as plt

from helpers.helpers import HelperFuncs

class PlotBldgDmg(HelperFuncs):
    def __init__(self):
        super().__init__()
        
    def plot(self, domain_size="estero", remove_elevated=False):
        bldgs = gpd.read_file(self.path_to_bldgs)
        dmg   =  pd.read_csv(self.path_to_dmg)
        if remove_elevated:
            remove_bldgs = (dmg["FFE_elev_status"] == "elevated") & (dmg["FFE_foundation"]=="Piles/Columns")
            dmg = dmg.loc[~remove_bldgs]
        
        bldgs = pd.merge(bldgs[["FolioID", "geometry"]], dmg[["TA_FolioID", "VDA_DS_overall"]], left_on="FolioID", right_on="TA_FolioID")
        

        xo, yo, theta = self.get_origin()
        bldgs["geometry"] = bldgs["geometry"].rotate(angle=-theta, origin=(xo, yo))

        figsize = self.get_figsize(domain_size)
        fig, ax = plt.subplots(1,1,figsize=figsize)
        bldgs.plot(ax=ax, column="VDA_DS_overall", cmap="RdYlGn_r", 
                    legend=True, 
                    legend_kwds={"ncols":1, 
                                "bbox_to_anchor":(1.05,1.01),
                                "loc":"upper left"})

        self.remove_frame(ax)
        self.save_fig(fig, "bldg_dmg",
                        # transparent=True,
                        dpi=2000,
                        )


