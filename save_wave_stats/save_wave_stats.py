import os
import math
import rasterio
import rasterstats
import numpy as np
import pandas as pd
import geopandas as gpd
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
                    if var == "zs1":            # if zs1, have water elevation time series, need to get H
                        H = self.get_H(z)       #   getting wave heights from time series
                    elif var == "H":            # else, can just use H since this is wave height from group.
                        H = z
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

        fn_out = os.path.join(self.path_to_save_plot, "Hs.tiff")
        Hs = self.read_npy(stat)
        bldgs = self.read_buildings()
        Hs_bldg = self.assign_max_to_bldgs(Hs, bldgs)
        Hs = np.fmax(Hs, Hs_bldg)
        self.create_rotated_raster(Hs, crs="epsg:32617", xo=xo, yo=yo, dx=dx, dy=dy,
                                   theta=theta, output_filepath=fn_out)

    def assign_to_bldgs(self, stat, path_to_bldgs, runs=None, col_names=None):
        bldgs = gpd.read_file(path_to_bldgs)
        if runs != None:
            runs.insert(0, self.model_runname)
        else:
            runs = [self.model_runname]

        bldgs["geometry"] = bldgs["geometry"].centroid
        coord_list = [(x, y) for x, y in zip(bldgs["geometry"].x, bldgs["geometry"].y)]
        if col_names==None:
            col_names = []
            need_to_create_col_names = True
        else:
            need_to_create_col_names = False

        cnt = 0
        for run in runs:
            rstr = os.path.join(self.path_to_save_plot, "..", run,  "{}.tiff" .format(stat))

            if need_to_create_col_names == True:
                col_name = "Hs_{}" .format(run)    
                col_names.append(col_name)
            else:
                col_name = col_names[cnt]

            with rasterio.open(rstr, "r") as r:
                bldgs[col_name] = [x[0] for x in r.sample(coord_list)]
            cnt += 1

        keep_cols = ["TARGET_FID", "OBJECTID", "FolioID"] + col_names
        bldgs = bldgs[keep_cols]
        fn_out = os.path.join(self.path_to_save_plot, "Hs_at_bldgs.csv")
        bldgs.to_csv(fn_out, index=False)

    # def create_rotated_raster2(self, H, crs, xo, yo, dx, dy, theta, output_filepath):
    #     """
    #     Creates a GeoTIFF raster from a NumPy array with rotation.

    #     Args:
    #         data_array (np.ndarray): The 2D array of data to write out.
    #         crs (str | dict): The Coordinate Reference System (e.g., 'EPSG:4326').
    #         xo (float): The x-coordinate of the origin (top-left corner).
    #         yo (float): The y-coordinate of the origin (top-left corner).
    #         theta (float): The counter-clockwise rotation angle in **degrees**.
    #         output_filepath (str): The path to save the output GeoTIFF file.
    #     """
    #     H = np.flipud(H)

    #     rows, cols = H.shape
    #     length = rows*dx
    #     width = cols*dy
        
    #     xo, yo = 407250.7012256911, 2924527.223300225


    #     # theta = 0
    #     theta_rad = np.deg2rad(theta)


    #     # --- 4. Calculate the Rotated Affine Transform ---
    #     # (a, b, d, e terms for rotation are unchanged)
    #     a = 1 * math.cos(theta_rad)
    #     b = 1 * math.sin(theta_rad)
    #     # c = 
    #     d = 1 * math.sin(theta_rad)
    #     e = -1 * math.cos(theta_rad) 
    #     # f = 

    #     # Create the Affine transform using the upper-left X (x_min) and Y (y_max)
    #     transform = rasterio.transform.Affine(
    #         a,  # a
    #         b,  # b
    #         xo,  # c: X-coordinate of the upper-left corner
    #         d,  # d
    #         e,  # e
    #         yo   # f: Y-coordinate of the upper-left corner
    #     )

    #     print(f"Calculated Affine Transform:\n{transform}")

    #     profile = {
    #         'driver': 'GTiff',
    #         'dtype': H.dtype,
    #         'nodata': -99999,
    #         'width': cols,
    #         'height': rows,
    #         'count': 1,
    #         'crs': crs,
    #         'transform': transform,
    #     }

    #     try:
    #         with rasterio.open(output_filepath, 'w', **profile) as dst:
    #             dst.write(H, 1)
    #         print(f"Successfully created raster at: {output_filepath}")
    #         print(f"Raster CRS: {crs}")
    #     except Exception as e:
    #         print(f"An error occurred during raster writing: {e}")




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























