import os
import math
from tqdm import tqdm
import rasterio
import numpy as np
import pandas as pd
import geopandas as gpd
from helpers.helpers import HelperFuncs
import concurrent
# from concurrent.futures import ThreadPoolExecutor
import threading

class SaveWaveStats(HelperFuncs):
    """docstring for plot_wave_heights"""
    def __init__(self):
        super().__init__()


    def _process_spatial_point(self, y2_, x2_, t, t_idxs, t_idx_start, sample_freq, data_all, store_in_mem, stats, chunk_size_sec, data_save_dict, data_save_dict_lock):
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
                data_ = np.max([self.compute_Hs(i) for i in H_chunks]).item()
            elif stat == "Hmax":
                try:
                    data_ = np.max(H).item() # Use .item() for consistency with scalar results
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
                data_ = np.max(z).item()

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


    def save(self, var, stats, trim_beginning_seconds=0, sample_freq=1, store_in_mem=False, chunk_size_min=15, max_workers=None):
        """
        Function to save wave stats to .npy files, parallelized over spatial points.
        
        max_workers: The maximum number of threads to use for parallel processing. 
                     If None, it defaults to the number of processors.
        """
        chunk_size_sec = chunk_size_min * 60
        t = self.read_time_xarray()
        t_idx_start = np.argmin(np.abs(t - trim_beginning_seconds))
        t = t[t_idx_start::sample_freq]  # time array with beginning trimmed off.

        # The rest of the setup is the same
        t_chunks_val = np.arange(t[0], t[-1], step=chunk_size_sec)
        t_idxs = [self.time_to_tindex(t_, t) for t_ in t_chunks_val]
        dims = self.read_dims_xarray()  # dimensions of grid
        # dims = (3,dims[1])

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
                        y2_, x2_, t, t_idxs, t_idx_start, sample_freq, 
                        data_all, store_in_mem, stats, chunk_size_sec, 
                        data_save_dict, data_save_dict_lock
                    )
                    futures.append(future)

            # Use tqdm to monitor the progress of the submitted tasks
            for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Processing spatial points"):
                # You can get the result if needed, but here it's mainly for monitoring
                future.result() 

        print(f"Finished processing {dims[0]*dims[1]} spatial points.")
        print("writing to {}:".format(self.path_to_save_plot))
        self.write_data_save_dict(data_save_dict, self.path_to_save_plot)

    # def save(self, var, stats, trim_beginning_seconds=0, sample_freq=1, store_in_mem=False, 
    #         chunk_size_min=15):
    #     """
    #         function to save wave stats to .npy files. these are used later
    #         var: variable that is written by XBeach
    #         stats: wave statistic to save. options include:
    #             Hs: 
    #             Hs_max: 
    #             Hs_tot: 
    #             Hmax: 
    #             t_Hs_Nm: 
    #             zsmax: 
    #             Tm: 
    #         trim_beginning_seconds: seconds to trim off of the beginning of the time series
    #         store_in_mem: boolean to store entire dataset in memory when running; not possbile for large XBeach runs
    #         chunk_size_min: chunk size in minutes; used to compute wave statistics at each tim interval
    #     """
    #     chunk_size_sec = chunk_size_min*60
    #     t = self.read_time_xarray()
    #     t_idx_start = np.argmin(np.abs(t-trim_beginning_seconds))
    #     t = t[t_idx_start::sample_freq]     # time array with beginning trimmed off.

    #     t_chunks = np.arange(t[0], t[-1], step=chunk_size_sec)  # time array corresponding to chunks
    #     t_idxs = [self.time_to_tindex(t_,t) for t_ in t_chunks] # indicies to chunk data
    #     dims = self.read_dims_xarray()      # dimensions of grid

    #     if store_in_mem:    # if storing the entire dataset in memory while processing
    #         data_all = self.read_3d_data_xarray(var)
    #     else:               # else, read as an xarray dataset
    #         data_all = self.read_3d_data_xarray_nonmem(var)

    #     data_save_dict = self.setup_data_save_dict(stats, t_idxs, dims) # dictionary to save all data
    #     for y2_ in range(dims[0]):
    #         print("y2_ = {} out of {}" .format(y2_, dims[0]))
            
    #         for x2_ in tqdm(range(dims[1])):
    #             if store_in_mem:
    #                 z = data_all[t_idx_start::sample_freq,y2_,x2_]
    #             else:
    #                 z = data_all[t_idx_start::sample_freq,y2_,x2_].values

    #             z_chunks = []
    #             t_chunks = []
    #             t_idx_prior = 0
    #             for t_idx in t_idxs[1:]:
    #                 z_chunks.append(z[t_idx_prior:t_idx])
    #                 t_chunks.append(t[t_idx_prior:t_idx])
    #                 t_idx_prior = t_idx
                
    #             H = self.get_H(z)
    #             H_chunks = [self.get_H(z_) for z_ in z_chunks]
    #             T = self.get_T(z, t)
    #             T_chunks = [self.get_T(z_chunks[i],t_chunks[i]) for i in range(len(z_chunks))]

    #             prnt=False
    #             for stat in stats:
    #                 # signficant wave height; one for each time chunk
    #                 if stat == "Hs":
    #                     data_ = [self.compute_Hs(i) for i in H_chunks]

    #                 # total signficant wave height across entire time series
    #                 elif stat == "Hs_tot":
    #                     data_ = self.compute_Hs(H)

    #                 # maximum signficant wave height from chunks
    #                 elif stat == "Hs_max":
    #                     data_ = np.max([self.compute_Hs(i) for i in H_chunks]).item()

    #                 # maximum wave height across entire record
    #                 elif stat == "Hmax":
    #                     try:
    #                         data_ = np.max(H)
    #                     except ValueError:
    #                         data_ = 0
    #                     # data_ = np.max(H)

    #                 # mean period
    #                 elif stat == "Tm":
    #                     data_ = []
    #                     for t in T_chunks:
    #                         try:
    #                             data_.append(np.mean(t))
    #                         except:
    #                             data_.append(0)
    #                     # data_ = [np.mean(i).item() for i in T_chunks]
                        
    #                 # time that signficant wave height exceeds threshold value
    #                 elif "t_Hs_" in stat:
    #                     val = stat.split("_")[-1]
    #                     val = float(val.split("m")[0])
    #                     Hs = np.array([self.compute_Hs(i) for i in H_chunks])
    #                     Hs_greater = Hs>val
    #                     data_ = np.sum(Hs_greater)*chunk_size_sec

    #                 # maximum water elevation in entire time series                   
    #                 elif stat == "zs_max":
    #                     data_ = np.max(z).item()
                    
    #                 if prnt:
    #                     if isinstance(data_, list):
    #                         print("{}: {}" .format(stat, data_))
    #                     else:
    #                         print("{}: {:0.3f}" .format(stat, data_))

    #                 # storing in dataframe
    #                 if isinstance(data_, list):
    #                     for cnt, key in enumerate(data_save_dict[stat].keys()):
    #                         data_save_dict[stat][key][y2_,x2_] = data_[cnt]
    #                 else:
    #                     data_save_dict[stat][y2_, x2_] = data_

    #     print("writing to {}:" .format(self.path_to_save_plot))
    #     self.write_data_save_dict(data_save_dict, self.path_to_save_plot)


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
    
    def geolocate(self, stat="Hs"):
        fn_params = os.path.join(self.path_to_model, "params.txt")
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
        bldgs = self.read_buildings()
        Hs_bldg = self.assign_max_to_bldgs(Hs, bldgs)
        Hs = np.fmax(Hs, Hs_bldg)
        self.create_rotated_raster(Hs, crs="epsg:32617", xo=xo, yo=yo, dx=dx, dy=dy,
                                   theta=theta, output_filepath=fn_out)

    def assign_to_bldgs(self, stats, path_to_bldgs, runs=None, col_names=None):
        bldgs = gpd.read_file(path_to_bldgs)
        
        if runs != None:
            runs.insert(0, self.model_runname)
        else:
            runs = [self.model_runname]

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

                rstr = os.path.join(self.path_to_save_plot, "..", run,  "{}.tiff" .format(stat))            
                with rasterio.open(rstr, "r") as r:
                    bldgs[col_name] = [x[0] for x in r.sample(coord_list)]

                cnt += 1

        dem = os.path.join(os.getcwd(), "..", "data", "dem", "dem-resampled.tiff")
        self.reproject_raster(dem, bldgs.crs)
        with rasterio.open("temp.tiff", "r") as r:
            bldgs["dem_elev"] = [x[0] for x in r.sample(coord_list)]
            os.remove("temp.tiff")
    
        # bldgs.to_crs("epsg:4326", inplace=True)
        bldgs["centroid"] = bldgs["centroid"].to_crs("epsg:4326")
        bldgs["lon"] = bldgs["centroid"].x
        bldgs["lat"] = bldgs["centroid"].y

        max_surge = 3.740
        bldgs["water_depth_temp"] = max_surge-bldgs["dem_elev"]

        keep_cols = ["VDA_id", "TARGET_FID", "OBJECTID", "FolioID", "lon", "lat", "water_depth_temp"] + col_names
        bldgs = bldgs[keep_cols]
        fn_out = os.path.join(self.path_to_save_plot, "H_at_bldgs.csv")
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
        a = np.cos(theta_rad)
        b = -np.sin(theta_rad)
        c = xo
        d = np.sin(theta_rad)
        e = np.cos(theta_rad)
        f = yo
        
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























