import os
import numpy as np
import xarray as xr



class HelperFuncs():
    """docstring for HelperFuncs"""
    def __init__(self, arg):
        self.arg = arg

    def read_data_xarray_pt(self, var, idx, idy, prnt_read=False, rtn_time_array=False):
        fn = os.path.join(self.path_to_model, "xboutput.nc")
        ds = xr.open_dataset(fn, chunks={"globaltime": 100})
        if prnt_read:
            print("Dataset object read:")
            print(ds)
            print("\n\n")
        
        slice_data = ds[var][:,idy,idx]

        if rtn_time_array:
            time = ds["globaltime"].values
            return slice_data.values, time
        else:
            return slice_data.values
