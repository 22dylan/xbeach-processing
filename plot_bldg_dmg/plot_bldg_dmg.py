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
        
    def plot(self, 
            domain_size="estero", 
            bldgs="non-elevated", 
            remove_DSs=None, 
            plot_elevated=True, 
            fname=None):
        bldg_df = gpd.read_file(self.path_to_bldgs)
        bldg_df.set_index("VDA_id", inplace=True)
        dmg   =  pd.read_csv(self.path_to_dmg)
        dmg.set_index("VDA_id", inplace=True)
        bldg_df = pd.merge(bldg_df, dmg, left_index=True, right_index=True)

        elevated_bldgs = (bldg_df["FFE_elev_status"] == "elevated") & (bldg_df["FFE_foundation"]=="Piles/Columns")
        if bldgs=="all":
            txt = "All buildings (including elevated)"
            edgecolor="none"
        elif bldgs=="non-elevated":
            bldg_df = bldg_df[~elevated_bldgs]
            txt = "Ignore Elevated"
            edgecolor="none"
        elif bldgs=="elevated":
            bldg_df = bldg_df[elevated_bldgs]
            txt = "Elevated Only"
            edgecolor='k'
        else:
            raise ValueError("bldgs keyword must be: `all`, `elevated` or `non-elevated`")

        xo, yo, theta = self.get_origin()
        bldg_df["geometry"] = bldg_df["geometry"].rotate(angle=-theta, origin=(xo, yo))
        
        figsize = self.get_figsize(domain_size)
        fig, ax = plt.subplots(1,1,figsize=figsize)
        if remove_DSs != None:
            cmap = mpl.colors.ListedColormap(["darkseagreen", "red"])

            bldg_df["removed"] = 0
            bldg_df.loc[bldg_df["VDA_DS_overall"].isin(remove_DSs), "removed"] = 1
            
            bldg_df.plot(
                    ax=ax, 
                    column="removed", 
                    categorical=True,
                    cmap=cmap,
                    legend=True,
                    edgecolor=edgecolor,
                    legend_kwds={
                                "labels":["Standing (DS0 to DS5)", "Destroyed (DS6)"],
                                "bbox_to_anchor":(1.05,1.01),
                                "loc":"upper left"}
                    )

            if bldgs=="all":
                # elevated_bldgs = (bldg_df["FFE_elev_status"] == "elevated") & (bldg_df["FFE_foundation"]=="Piles/Columns")                
                bldgs_elevated = bldg_df[elevated_bldgs]
                bldgs_elevated.plot(
                    ax=ax,
                    color="none",
                    edgecolor='k',
                    legend_kwds={"labels":["Elevated"]}
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


