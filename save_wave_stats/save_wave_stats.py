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

from typing import SupportsIndex
import numpy as np
from numpy._typing._array_like import _ArrayLikeComplex_co, _ArrayLikeTD64_co, _ArrayLikeObject_co

class SaveWaveStats(HelperFuncs):
    """docstring for plot_wave_heights"""
    def __init__(self):
        super().__init__()
        self.rho = 1025     # density of salt (kg/m^3)
        self.g = 9.81       # gravity (m/s^2)

    def save_removed_bldgs(self):
        bldgs = self.read_removed_bldgs()
        fn_out = os.path.join(self.path_to_save_plot, "removed_bldgs.npy")
        np.save(fn_out, bldgs)


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

        if any(["velocity" in i for i in stats]):
            self.ue_data = self.read_3d_data_xarray_nonmem("ue")
            self.ve_data = self.read_3d_data_xarray_nonmem("ve")

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
            elif stat == "velocity_magnitude":
                ue = self.ue_data[t_idx_start::sample_freq, y2_, x2_].values
                ve = self.ve_data[t_idx_start::sample_freq, y2_, x2_].values
                data_ = self.compute_velocity_mag(ue, ve)

            elif stat == "velocity_direction":
                ue = self.ue_data[t_idx_start::sample_freq, y2_, x2_].values
                ve = self.ve_data[t_idx_start::sample_freq, y2_, x2_].values
                data_ = self.compute_velocity_dir(ue, ve)

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

    def compute_velocity_mag(self, ue, ve):        
        mag = np.sqrt(np.square(ue) + np.square(ve))
        if np.all(np.isnan(mag)):
            return np.nan
        return np.nanmax(mag).item()

    def compute_velocity_dir(self, ue, ve):
        mag = np.sqrt(np.square(ue) + np.square(ve))
        if np.all(np.isnan(mag)):
            return np.nan
        max_idx = np.nanargmax(np.sqrt(np.square(ue) + np.square(ve))).item()
        ue, ve = ue[max_idx], ve[max_idx]
        return self.compute_angle(ue, ve)

    def compute_angle(self, x_vec, y_vec):
        """
        Calculates the angle of a 2D vector from the x-axis in a counter-clockwise direction
        and returns it in the range [0, 360) degrees.
        """
        angles_rad = np.arctan2(y_vec, x_vec)
        angles_deg = np.degrees(angles_rad)
        # Convert negative angles to the [0, 360) range
        if angles_deg < 0:
            angles_deg += 360
        # angles_deg[angles_deg < 0] += 360
        return angles_deg

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

    def geolocate(self, stat="Hs"):
        fn_params = os.path.join(self.path_to_model, "params.txt")
        if os.path.exists(fn_params):
            model_dir = self.path_to_model
        else:
            hs0 = self.set_hotstart_runs()[0]
            model_dir = os.path.join(self.path_to_model, hs0)
            fn_params = os.path.join(model_dir, "params.txt")

        with open(fn_params,'r') as f:
            for cnt, line in enumerate(f.readlines()):
                if "xo" in line:
                    l_ = [i.strip() for i in line.split()]
                    xo = float(l_[-1])
                if "yo" in line:
                    l_ = [i.strip() for i in line.split()]
                    yo = float(l_[-1])
                if "theta" in line:
                    l_ = [i.strip() for i in line.split()]
                    theta = float(l_[-1])
                if "dx" in line:
                    if "vardx" in line:
                        continue
                    l_ = [i.strip() for i in line.split()]
                    dx = float(l_[-1])
                if "dy" in line:
                    l_ = [i.strip() for i in line.split()]
                    dy = float(l_[-1])

        
        fn_out = os.path.join(self.path_to_save_plot, "{}.tiff" .format(stat))
        Hs = self.read_npy(stat)
        bldgs = self.read_buildings(model_dir)
        Hs_bldg = self.assign_max_to_bldgs(Hs, bldgs)
        Hs = np.fmax(Hs, Hs_bldg)
        self.create_rotated_raster(Hs, crs="epsg:32617", xo=xo, yo=yo, dx=dx, dy=dy,
                                   theta=theta, output_filepath=fn_out)

    def assign_to_bldgs(self, stats, runs=None, col_names=None, fname=None):
        bldgs = gpd.read_file(self.path_to_bldgs)
        
        if runs != None:
            runs.insert(0, self.path_to_save_plot)
        else:
            runs = [self.path_to_save_plot]
        runs = list(filter(None, runs))

        bldgs["centroid"] = bldgs["geometry"].centroid
        coord_list = [(x, y) for x, y in zip(bldgs["centroid"].x, bldgs["centroid"].y)]
        if col_names==None:
            col_names = []
            need_to_create_col_names = True
        else:
            need_to_create_col_names = False

        cnt = 0
        for stat in stats:
            for run in runs:
                if need_to_create_col_names == True:
                    col_name = "{}_{}" .format(stat, run)    
                    col_names.append(col_name)
                else:
                    col_name = col_names[cnt]

                rstr = os.path.join(self.path_to_save_plot, "..", run, "{}.tiff" .format(stat))
                with rasterio.open(rstr, "r") as r:
                    bldgs[col_name] = [x[0] for x in r.sample(coord_list)]

                cnt += 1

        bldgs["centroid"] = bldgs["centroid"].to_crs("epsg:4326")
        bldgs["lon"] = bldgs["centroid"].x
        bldgs["lat"] = bldgs["centroid"].y

        keep_cols = ["VDA_id", "TARGET_FID", "OBJECTID", "FolioID", "lon", "lat"] + col_names
        bldgs = bldgs[keep_cols]
        if fname == None:
            fname = "H_at_bldgs.csv"
        fn_out = os.path.join(self.path_to_save_plot, fname)
        bldgs.to_csv(fn_out, index=False)


    def create_rotated_raster(self, H, crs, xo, yo, dx, dy, theta, output_filepath):
        """
        Creates a GeoTIFF raster from a NumPy array with rotation.

        Args:
            data_array (np.ndarray): The 2D array of data to write out.
            crs (str | dict): The Coordinate Reference System (e.g., 'EPSG:4326').
            xo (float): The x-coordinate of the origin (top-left corner).
            yo (float): The y-coordinate of the origin (top-left corner).
            theta (float): The counter-clockwise rotation angle in **degrees**.
            output_filepath (str): The path to save the output GeoTIFF file.
        """
        rows, cols = H.shape


        theta_rad = np.deg2rad(theta)
        # -- old
        a = np.cos(theta_rad)
        b = -np.sin(theta_rad)
        c = xo
        d = np.sin(theta_rad)
        e = np.cos(theta_rad)
        f = yo
        # -- old

        # -- new
        a = dx * np.cos(theta_rad)  # x-scale * cos(theta)
        b = -dx * np.sin(theta_rad) # y-shear * -sin(theta) (Note: dx is usually used here for non-square pixels)
        c = xo                      # x-translation (xo)
        d = dx * np.sin(theta_rad)  # x-shear * sin(theta)
        e = dy * np.cos(theta_rad) # y-scale * cos(theta) (Note: -dy is used for the standard top-left origin)
        f = yo                      # y-translation (yo)
        # -- new

        # Create the transform
        transform = rasterio.transform.Affine(a, b, c, d, e, f)

        print(f"Calculated Affine Transform:\n{transform}")

        # 3. Define the raster metadata
        profile = {
            'driver': 'GTiff',
            'dtype': H.dtype,
            'nodata': -9999, # Example NoData value, adjust as needed
            'height': rows,
            'width': cols,
            'count': 1, # Number of bands
            'crs': crs,
            'transform': transform,
            # Compression and tiling options can be added here
            'tiled': True,
            'compress': 'lzw',
        }

        # 4. Write the array to a GeoTIFF file
        try:
            with rasterio.open(output_filepath, 'w', **profile) as dst:
                # Write the data. We write to band 1 (index 1)
                dst.write(H, 1)
            print(f"Successfully created raster at: {output_filepath}")
            print(f"Raster CRS: {crs}")
        except Exception as e:
            print(f"An error occurred during raster writing: {e}")



    def get_offset_mask(self, labeled_mask, i):
        m_ = ~(labeled_mask==i)
        m_ = np.pad(m_, pad_width=1, mode="constant", constant_values=True)
        shifted_up = m_[2:, 1:-1]
        shifted_down = m_[:-2, 1:-1]
        shifted_left = m_[1:-1, 2:]
        shifted_right = m_[1:-1, :-2]
        original_mask_trimmed = m_[1:-1, 1:-1]

        offset_mask = original_mask_trimmed & shifted_up & shifted_down & shifted_left & shifted_right
        return ~offset_mask

