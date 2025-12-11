import os
import numpy as np
import pandas as pd
import geopandas as gpd

import matplotlib as mpl
import matplotlib.pyplot as plt

from helpers.helpers import HelperFuncs

class PlotBldgDmg(HelperFuncs):
    def __init__(self):
        super().__init__()
        
    def plot(self, domain_size="estero", remove_elevated=False, remove_DSs=None, fname=None):
        bldgs = gpd.read_file(self.path_to_bldgs)
        dmg   =  pd.read_csv(self.path_to_dmg)
        if remove_elevated:
            remove_bldgs = (dmg["FFE_elev_status"] == "elevated") & (dmg["FFE_foundation"]=="Piles/Columns")
            dmg = dmg.loc[~remove_bldgs]
        
        bldgs = pd.merge(bldgs[["FolioID", "geometry"]], dmg[["TA_FolioID", "VDA_DS_overall"]], left_on="FolioID", right_on="TA_FolioID")
        
        fn_params = os.path.join(self.path_to_model, "params.txt")
        if os.path.exists(fn_params):
            model_dir = self.path_to_model
        else:
            hs0 = self.set_hotstart_runs()[0]
            model_dir = os.path.join(self.path_to_model, hs0)
            fn_params = os.path.join(model_dir, "params.txt")

        xo, yo, theta = self.get_origin(model_dir=model_dir)
        bldgs["geometry"] = bldgs["geometry"].rotate(angle=-theta, origin=(xo, yo))
        
        figsize = self.get_figsize(domain_size)
        fig, ax = plt.subplots(1,1,figsize=figsize)
        if remove_DSs != None:
            cmap = mpl.colors.ListedColormap(["darkseagreen", "red"])

            bldgs["removed"] = 0
            bldgs.loc[bldgs["VDA_DS_overall"].isin(remove_DSs), "removed"] = 1
            bldgs.plot(
                    ax=ax, 
                    column="removed", 
                    categorical=True,
                    cmap=cmap,
                    legend=True,
                    # 
                    legend_kwds={
                                "labels":["Standing (DS0 to DS5)", "Destroyed (DS6)"],
                                "bbox_to_anchor":(1.05,1.01),
                                "loc":"upper left"}
                    )
        else:
            bldgs.plot(ax=ax, column="VDA_DS_overall", cmap="RdYlGn_r", 
                        legend=True, 
                        legend_kwds={"ncols":1, 
                                    "bbox_to_anchor":(1.05,1.01),
                                    "loc":"upper left"})

        self.remove_frame(ax)
        self.save_fig(fig, 
                      fname,
                      # transparent=True,
                      dpi=2000,
                      )


