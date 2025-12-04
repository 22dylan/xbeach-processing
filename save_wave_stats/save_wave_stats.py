import os
import math
from tqdm import tqdm
import rasterio
import numpy as np
import pandas as pd
import geopandas as gpd
from helpers.helpers import HelperFuncs
import concurrent
import threading

import scipy.ndimage as ndi

# testing
# from __future__ import annotations
from typing import SupportsIndex
import numpy as np
from numpy._typing._array_like import _ArrayLikeComplex_co, _ArrayLikeTD64_co, _ArrayLikeObject_co

class SaveWaveStats(HelperFuncs):
    """docstring for plot_wave_heights"""
    def __init__(self):
        super().__init__()
        self.rho = 1025     # density of salt (kg/m^3)
        self.g = 9.81       # gravity (m/s^2)

    def save(self, var, stats, trim_beginning_seconds=0, sample_freq=1, 
            store_in_mem=False, chunk_size_min=15, avg_window_min=2, max_workers=None):
        """
        Function to save wave stats to .npy files, parallelized over spatial points.
        
        max_workers: The maximum number of threads to use for parallel processing. 
                     If None, it defaults to the number of processors.
        """
        chunk_size_sec = chunk_size_min * 60
        t = self.read_time_xarray()
        t_idx_start = np.argmin(np.abs(t - trim_beginning_seconds))
        t = t[t_idx_start::sample_freq]  # time array with beginning trimmed off.
        avg_window_sec = avg_window_min*60

        # The rest of the setup is the same
        t_chunks_val = np.arange(t[0], t[-1], step=chunk_size_sec)
        t_idxs = [self.time_to_tindex(t_, t) for t_ in t_chunks_val]
        dims = self.read_dims_xarray()  # dimensions of grid

        # Read data outside the loop
        if store_in_mem:
            data_all = self.read_3d_data_xarray(var)
        else:
            data_all = self.read_3d_data_xarray_nonmem(var)

        data_save_dict = self.setup_data_save_dict(stats, t_idxs, dims)
        
        # Create a lock for thread-safe access to the shared data_save_dict
        data_save_dict_lock = threading.Lock() # Import threading at the top

        print(f"Starting parallel processing with {max_workers if max_workers else 'default'} workers...")

        # Parallel Execution over spatial dimensions
        futures = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            for y2_ in range(dims[0]):
                for x2_ in range(dims[1]):
                    # Submit the processing of a single spatial point as a task
                    future = executor.submit(
                        self._process_spatial_point, 
                        y2_, x2_, t, t_idxs, t_idx_start, sample_freq, avg_window_sec,
                        data_all, store_in_mem, stats, chunk_size_sec, 
                        data_save_dict, data_save_dict_lock
                    )
                    futures.append(future)
            # all tasks have been submitted, now using tqdm to monitor the progress
            for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Processing spatial points"):
                future.result() 

        print(f"Finished processing {dims[0]*dims[1]} spatial points.")
        print("writing to {}:".format(self.path_to_save_plot))
        self.write_data_save_dict(data_save_dict, self.path_to_save_plot)

    def _process_spatial_point(self, y2_, x2_, t, t_idxs, t_idx_start, sample_freq, avg_window_sec,
                            data_all, store_in_mem, stats, chunk_size_sec, data_save_dict, 
                            data_save_dict_lock):
        """
        Helper function to process a single (y, x) spatial point.
        This function will be executed in parallel.
        """
        if store_in_mem:
            z = data_all[t_idx_start::sample_freq, y2_, x2_]
        else:
            z = data_all[t_idx_start::sample_freq, y2_, x2_].values

        z_chunks = []
        t_chunks = []
        t_idx_prior = 0
        for t_idx in t_idxs[1:]:
            z_chunks.append(z[t_idx_prior:t_idx])
            t_chunks.append(t[t_idx_prior:t_idx])
            t_idx_prior = t_idx

        H = self.get_H(z)
        H_chunks = [self.get_H(z_) for z_ in z_chunks]
        T_chunks = [self.get_T(z_chunks[i], t_chunks[i]) for i in range(len(z_chunks))]
        dt = t[1] - t[0]    # time step
        
        # Dictionary to hold the results for this single spatial point
        point_results = {}
        
        # compute statistics
        for stat in stats:
            data_ = None # Reset for each stat

            if stat == "Hs":
                data_ = [self.compute_Hs(i) for i in H_chunks]
            elif stat == "Hs_tot":
                data_ = self.compute_Hs(H)
            elif stat == "Hs_max":
                data_ = np.nanmax([self.compute_Hs(i) for i in H_chunks]).item()
            elif stat == "Hmax":
                try:
                    data_ = np.nanmax(H).item() # Use .item() for consistency with scalar results
                except ValueError:
                    data_ = 0
            elif stat == "Tm":
                data_ = []
                for t_chunk in T_chunks:
                    try:
                        data_.append(np.mean(t_chunk).item()) # Use .item()
                    except:
                        data_.append(0)
            elif "t_Hs_" in stat:
                val = stat.split("_")[-1]
                val = float(val.split("m")[0])
                Hs = np.array([self.compute_Hs(i) for i in H_chunks])
                Hs_greater = Hs > val
                data_ = np.sum(Hs_greater) * chunk_size_sec
            elif stat == "zs_max":
                data_ = np.nanmax(z).item()
            elif stat == "zs_mean":
                data_ = np.nanmean(z).item()
            elif stat == "surge_max":
                data_ = np.nanmax([np.nanmean(i) for i in z_chunks]).item()
            elif stat == "impulse":
                data_ = self.compute_impulse(z, t, dt, avg_window_sec)

            if data_ is not None:
                point_results[stat] = data_
                
        # 5. Lock and store results into the shared dictionary
        # In a parallel environment, writing to a shared resource (data_save_dict)
        # must be protected to prevent race conditions.
        with data_save_dict_lock:
            for stat, data_ in point_results.items():
                if isinstance(data_, list):
                    # Data that is a list (e.g., Hs, Tm) is saved by time chunk
                    for cnt, key in enumerate(data_save_dict[stat].keys()):
                        data_save_dict[stat][key][y2_, x2_] = data_[cnt]
                else:
                    # Scalar data (e.g., Hs_tot, Hmax, zs_max)
                    data_save_dict[stat][y2_, x2_] = data_

        return y2_, x2_ # Return for tracking/tqdm update

    def compute_impulse(self, z, t, dt, avg_window_sec):
        h, z_trimmed, time_trimmed = self.running_mean(z,t,avg_window_sec)
        eta = z_trimmed - h

        # calculate wave force
        fw = ((self.rho*self.g)/2)* np.abs((2*h*eta) + (np.square(eta)))  # units are N/m
        I = self.nantrapz(fw, dx=dt, axis=0)     # units are (N/m)-s
        # f = f*res                           # units are now N-s
        I = I/3600                          # units are now (N-hr)/m
        I = I/1000                          # units are now (kN-hr)/m
        return I

    def nantrapz(self,
        y: _ArrayLikeComplex_co | _ArrayLikeTD64_co | _ArrayLikeObject_co,
        x: _ArrayLikeComplex_co | _ArrayLikeTD64_co | _ArrayLikeObject_co | None = None,
        dx: float = 1.0,
        axis: SupportsIndex = -1,
    ):
        y = np.asanyarray(y)
        if x is None:
            d = dx
        else:
            x = np.asanyarray(x)
            if x.ndim == 1:
                d = np.diff(x)
                # reshape to correct shape
                shape = [1] * y.ndim
                shape[axis] = d.shape[0]
                d = d.reshape(shape)
            else:
                d = np.diff(x, axis=axis)
        nd = y.ndim
        slice1 = [slice(None)] * nd
        slice2 = [slice(None)] * nd
        slice1[axis] = slice(1, None)
        slice2[axis] = slice(None, -1)
        try:
            ret = np.nansum(d * (y[tuple(slice1)] + y[tuple(slice2)]) / 2.0, axis=axis)
        except ValueError:
            # Operations didn't work, cast to ndarray
            d = np.asarray(d)
            y = np.asarray(y)
            ret = np.add.reduce(d * (y[tuple(slice1)] + y[tuple(slice2)]) / 2.0, axis)
        return ret

    def setup_data_save_dict(self, stats, t_idxs, dims):
        data_save_dict = {}
        for stat in stats:
            if (stat == "Hs") or (stat == "Tm"):
                data_save_dict[stat] = {}
                for t_ in t_idxs[1:]:
                    data_save_dict[stat][t_] = np.empty(dims)
            else:
                data_save_dict[stat] = np.empty(dims)
        return data_save_dict

    def write_data_save_dict(self, data, output_dir=".", current_parts=None):
        """
        Recursively loops through a dictionary and saves all numpy arrays 
        to .npy files, using os.path.join for platform-agnostic path construction.

        Args:
            data (dict): The dictionary to process.
            output_dir (str): The base directory where files will be saved. Defaults to current directory (.).
            current_parts (list): Internal list of keys used to construct the nested filename.
        """
        if current_parts is None:
            current_parts = []
            
        # Ensure the output directory exists
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        for key, value in data.items():
            # Add the current key to the path parts
            new_parts = current_parts + [str(key)]
            base_name = "_".join(new_parts)
            if isinstance(value, np.ndarray):
                filename = base_name + ".npy"
                full_path = os.path.join(output_dir, filename)
                np.save(full_path, value)
                print("  Saved {}" .format(base_name))            
            elif isinstance(value, dict):
                self.write_data_save_dict(value, output_dir, new_parts)                
            
            else:
                print(f"Skipping non-array/non-dict item: {base_name} (Type: {type(value)})")

    def running_mean(self, z, t, N):
        cumsum = np.cumsum(np.insert(z,0,0)) # insert row of zeros at top
        h = (cumsum[N:] - cumsum[:-N]) / float(N)      # running mean with window size N; i.e. water depth, h
        
        # trimming up elevation and time data to be same length as running mean
        trim_start = (N - 1) // 2
        trim_end = trim_start + len(h)
        time_trimmed = t[trim_start:trim_end]
        z_trimmed = z[trim_start:trim_end]
        return h, z_trimmed, time_trimmed





















