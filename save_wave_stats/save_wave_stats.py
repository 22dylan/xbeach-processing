import os
import numpy as np
from helpers.helpers import HelperFuncs

class SaveWaveStats(HelperFuncs):
    """docstring for plot_wave_heights"""
    def __init__(self):
        super().__init__()

    def save(self, var, stat, store_in_mem=True):
        dims = self.read_dims_xarray()
        data_save = np.empty(dims)
        if store_in_mem:
            data_all = self.read_3d_data_xarray(var)
        else:
            data_all = self.read_3d_data_xarray_nonmem(var)

        for y2_ in range(dims[0]):
            print("y2_ = {} out of {}" .format(y2_, dims[0]))
            for x2_ in range(dims[1]):
                if store_in_mem:
                    z = data_all[:,y2_,x2_]
                else:
                    z = data_all[:,y2_,x2_].values

                if np.sum(z) == 0:
                    data_ = 0
                else:
                    H = self.get_H(z)       # getting wave heights from time series
                    if len(H) == 0:
                        data_ = 0
                    elif stat == "Hmax":
                        data_ = np.max(H)
                    elif stat == "Hs":
                        data_ = self.compute_Hs(H)
                data_save[y2_, x2_] = data_

        fn_out = os.path.join(self.path_to_save_plot, stat)
        np.save(fn_out, data_save)
        print("max wave height saved as: {}.npy" .format(fn_out))


