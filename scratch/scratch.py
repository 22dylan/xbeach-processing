import os
import matplotlib.pyplot as plt

from helpers.helpers import HelperFuncs

class Scratch(HelperFuncs):
    def __init__(self):
        super().__init__()


    def test(self):
        t = self.read_time_xarray()
        t_idx = self.time_to_tindex(time_wanted=14400, time=t)
        
        xgr, ygr, zgr = self.read_grid()
        idx, idy = self.xy_to_grid_index(xgr, ygr, xy=[490,524])
        # idx, idy = self.xy_to_grid_index(xgr, ygr, xy=[500,500])
        uu = self.read_pt_data_xarray(var="uu", idx=idx, idy=idy)
        # t_avg, uu_avg = self.calculate_running_avg(t, uu, window_sec=1000)
        
        
        fig, ax = plt.subplots(1,1, figsize=(10,6))
        ax.plot(t, uu)
        # ax.plot(t_avg, uu_avg, color='r', lw=2)
        

if __name__ == "__main__":
    pass