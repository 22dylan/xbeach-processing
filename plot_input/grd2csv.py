import os
import pandas as pd
import numpy as np
import rasterio
import affine

class grd2csv():
    """docstring for grd2csv"""
    def __init__(self, runname, rotate=True):
        model_dir = os.path.join(os.getcwd(), "..", "..", "models", runname)
        fn_xgrd = os.path.join(model_dir, "x.grd")
        fn_ygrd = os.path.join(model_dir, "y.grd")
        fn_zgrd = os.path.join(model_dir, "z.grd")

        xgrid = np.genfromtxt(fn_xgrd, delimiter=" ")
        ygrid = np.genfromtxt(fn_ygrd, delimiter=" ")
        zgrid = np.genfromtxt(fn_zgrd, delimiter=" ")
        zgrid = np.flipud(zgrid)

        self.save_raster_to_geotiff("temp.tiff", zgrid, xgrid, ygrid, rotate)

    def save_raster_to_geotiff(self, filename, raster_data, grid_x, grid_y, rotate=True):
        """
        Saves the raster data to a GeoTIFF file.

        Args:
            filename (str): The name of the output GeoTIFF file.
            raster_data (np.ndarray): The 2D array of interpolated z-values.
            grid_x (np.ndarray): The x-coordinates of the grid.
            grid_y (np.ndarray): The y-coordinates of the grid.
        """
        # Get the dimensions and data type from the raster data.
        height, width = raster_data.shape

        x_origin = 401474.7
        y_origin = 2925268.8
        angle_deg = 55.92839019260679
        x_rotation = -np.sin(np.deg2rad(55))
        y_rotation = np.sin(np.deg2rad(55))
        
        # Calculate the pixel size (resolution).
        pixel_width = (grid_x.max() - grid_x.min()) / (width - 1)
        pixel_height = (grid_y.max() - grid_y.min()) / (height - 1)

        # Create a geotransform, which tells GIS software where the raster is located.
        # The from_origin function takes: (origin_x, origin_y, pixel_width, pixel_height).
        # The origin is the top-left corner.
        transform = rasterio.transform.from_origin(x_origin, y_origin, pixel_width, pixel_height)
        if rotate == True:
            transform = (
                affine.Affine.translation(x_origin, y_origin) *
                affine.Affine.rotation(angle_deg) *
                affine.Affine.translation(-x_origin, -y_origin) *
                transform
            )
        


        print(transform)



        # Open a new GeoTIFF file in write mode ('w').
        with rasterio.open(
            filename,
            'w',
            driver='GTiff',         # Specify the GeoTIFF driver
            height=height,
            width=width,
            count=1,                # Number of bands (z-values)
            dtype=raster_data.dtype, # Data type of the raster
            transform=transform,     # The geotransform we just created
            nodata=np.nan            # Value for no data (NaNs in our case)
        ) as dst:
            # Write the raster data to the file, band 1.
            dst.write(raster_data, 1)
        
        print(f"Raster successfully saved to {filename}")


if __name__ == "__main__":
    g2c = grd2csv(runname="run6-bldgs", rotate=False)
