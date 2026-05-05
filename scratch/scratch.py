import os
import numpy as np
import pandas as pd

from helpers.helpers import HelperFuncs
from save_wave_stats.save_wave_stats import SaveWaveStats


class Scratch(HelperFuncs):
    def __init__(self):
        super().__init__()

    def run(self):
        gdf = self.read_bldgs_geodataframe()
        # duplicate_values = gdf.index[gdf.index.duplicated()].unique()
        
        ffe = pd.read_csv(self.path_to_dmg)
        ffe.set_index("VDA_id", inplace=True)
        print(len(gdf), len(ffe))
        missing_indices = ffe.index.difference(gdf.index)
        print(missing_indices)

        fds

        print(len(ffe))

        elevated, non_elevated = self.get_elevated_bldgs(gdf)
        
        print(len(elevated))
        print(len(non_elevated))

        

if __name__ == "__main__":
    pass
