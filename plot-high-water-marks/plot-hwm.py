import os
import numpy as np
import pandas as pd
import geopandas as gpd
import scipy.stats as st
import xarray as xr
import matplotlib.pyplot as plt

class plot_hwm():
    """docstring for plot_wave_heights"""
    def __init__(self, var="H"):
        self.file_dir = os.path.dirname(os.path.realpath(__file__))
        self.path_to_model = os.path.join(self.file_dir, "..", "..", "xbeach", "models")
        self.var = var

    def plot_scatter(self,model_runname, domain="regular", d_threshold=5, fname=None):
        model_dir = os.path.join(self.path_to_model, model_runname)
        max_zs = self.read_data_xarray_max(model_dir, var="zs")
        gdf_zs = self.max_zs_to_gdf(domain, max_zs)
        gdf_hwm = self.read_hwm()
        gdf_hwm.to_crs(gdf_zs.crs, inplace=True)
        gdf_comp = gpd.sjoin_nearest(gdf_hwm, gdf_zs[["zs", "geometry"]], how="left", distance_col="distance")
        gdf_comp = gdf_comp.loc[gdf_comp["distance"]<d_threshold]
        gdf_comp["elev_m"] = gdf_comp["elev_ft"]*0.3048

        # -- now plotting
        hwm = gdf_comp["elev_m"].values
        xbz = gdf_comp["zs"].values

        # setting up mask; ignore all NaN's and cells that are considered water.
        mask = ~np.isnan(hwm) & ~np.isnan(xbz)
        hwm_nonan, xbz_nonan = hwm[mask], xbz[mask]
        hwm_nonan = hwm_nonan.flatten()
        xbz_nonan = xbz_nonan.flatten()

        fig, ax = plt.subplots(1,1, figsize=(5,4))
        ax.scatter(xbz_nonan, hwm_nonan, facecolor="none", edgecolor="cadetblue",lw=1, s=10, zorder=0)

        # regression to r^2 and best fit line
        # slope, intercept, r_value, p_value, std_err = st.linregress(xbz_nonan, hwm_nonan)
        xbz_nonan = xbz_nonan[:,np.newaxis]
        slope, r_value, _, _ = np.linalg.lstsq(xbz_nonan, hwm_nonan)
        slope = slope[0]
        intercept = 0
        r_value = r_value[0]
        x = np.linspace(0,6, 100)
        y = slope*x + intercept
        ax.plot(x,y, ls="-", lw=1.5, color="purple", label="Regression")

        s1 = "Slope = {:0.4f}\n" .format(slope)
        s2 = "Intercept = {:0.4f}\n" .format(intercept)
        s3 = r"$r^2= $ {:0.4f}" .format(r_value)
        s = s1+s2+s3

        ax.text(x=0.95, y=0.05, s=s, 
                transform=ax.transAxes, 
                horizontalalignment='right', 
                verticalalignment="bottom",
                bbox=dict(facecolor='none', edgecolor='k'))

        # drawing 1 to 1 line
        ax.plot([-1,6], [-1,6], ls="-.", lw=1.0, zorder=1, color='k', label="1-to-1")

        ax.set_xlabel("XBeach")
        ax.set_ylabel("Observed")
        ax.set_xlim([2.5,5])
        ax.set_ylim([2.5,5])
        ax.legend(loc="upper left")
        ax.set_title("Water Elevation")

        if fname!=None:
            plt.savefig(fname,
                        transparent=False, 
                        dpi=300,
                        bbox_inches='tight',
                        pad_inches=0.1,
                        )
            plt.close()


    def max_zs_to_gdf(self, domain, max_zs):
        domain_dir = os.path.join(self.file_dir, "..", "..", "data", "xbeach-domain")
        if domain == "regular":
            fn = os.path.join(domain_dir, "xbeach-domain-epsg32617.geojson")
        domain = gpd.read_file(fn)
        (ny, nx) = np.shape(max_zs)
        
        x = domain.geometry.exterior[0].xy[0]
        y = domain.geometry.exterior[0].xy[1]

        l, w = [], []           # used to store both lengths and widths
        xo, yo = np.inf, np.inf         # x/y used for origin
        xa, ya = 0, 0                   # x/y used for calculating angle
        sides = []
        for i in range(len(x)):
            if i == 4:
                break
            # getting two points to calculate length/width
            x0, x1 = x[i], x[i+1]
            y0, y1 = y[i], y[i+1]
            dx = x[i+1] - x[i]
            dy = y[i+1] - y[i]
            d = np.sqrt(np.abs(dx)**2 + np.abs(dy)**2)
            sides.append(d)     # appending side length

            if y0<yo:       # getting origin point; crudely done. 
                xo, yo = x0, y0
            
            if x0>xa:       # getting point for calculating angle; crudely done
                xa, ya = x0, y0
        
        sides = np.array(sides)
        
        # note: the two lines below are used for large domain.
        w = sides[np.argsort(sides)[0:2]]       # width  is defined here as crossshore distance
        l = sides[np.argsort(sides)[2:]]        # length is defined here as alongshore distance

        theta_r, theta_d = self.compute_theta((xo, yo), (xa, ya))   # angle that grid is rotated, measured relative to east
        dx = np.round((w)/nx).mean()
        dy = np.round((l)/ny).mean()

        grid = np.zeros((nx,ny))
        pt_x, pt_y, pt_x_wrld, pt_y_wrld, idx, idy, zs = [], [], [], [], [], [], []
        for x in range(nx):
            for y in range(ny):
                pt_x.append(x*dx)
                pt_y.append(y*dy)

                pt_x_wrld_ = xo + pt_x[-1]*np.cos(theta_r) - pt_y[-1]*np.sin(theta_r)
                pt_y_wrld_ = yo + pt_x[-1]*np.sin(theta_r) + pt_y[-1]*np.cos(theta_r)
                pt_x_wrld.append(pt_x_wrld_)
                pt_y_wrld.append(pt_y_wrld_)
                idx.append(x)
                idy.append(y)
                zs.append(max_zs[y,x])


        grid_df = pd.DataFrame()
        grid_df["pt_x"] = pt_x
        grid_df["pt_y"] = pt_y
        grid_df["pt_x_wrld"] = pt_x_wrld
        grid_df["pt_y_wrld"] = pt_y_wrld
        grid_df["idx"] = idx
        grid_df["idy"] = idy
        grid_df["zs"] = zs        
        grid_gdf = gpd.GeoDataFrame(grid_df, geometry=gpd.points_from_xy(grid_df["pt_x_wrld"], grid_df["pt_y_wrld"]), crs="EPSG:32617")
        return grid_gdf

    def read_hwm(self):
        hwm_dir = os.path.join(self.file_dir, "..", "..", "data", "validation", "high-water-marks")
        fn1 = os.path.join(hwm_dir, "FilteredHWMs_extract_DEMs.csv")
        df1 = pd.read_csv(fn1)
        df1 = df1[["longitude", "latitude", "elev_ft"]]
        df1["source"] = "nathan-m"
        fn2 = os.path.join(hwm_dir, "Ian.high.water.marks.txt")
        headers = ["id", "elev_ft", "latitude", "longitude"]

        arr = np.genfromtxt(fn2)
        df2 = pd.DataFrame(arr, columns=headers)
        df2.set_index("id", inplace=True)
        df2["source"] = "don-s"

        df = pd.concat([df1, df2], ignore_index=True)
        gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df["longitude"], df["latitude"]), crs="EPSG:4326")
        del gdf["latitude"]
        del gdf["longitude"]

        return gdf

    def read_data_xarray_max(self, model_dir, var, prnt_read=False):
        fn = os.path.join(model_dir, "xboutput.nc")

        ds = xr.open_dataset(fn, chunks={"globaltime": 100})
        if prnt_read:
            print("Dataset object read:")
            print(ds)
            print("\n\n")
        
        max_vals = ds[var].max(dim="globaltime").values[:,:]
        return max_vals
    
    def myround(self, x, base=5):
        return base * round(x/base)

    def compute_theta(self, pt1, pt2):
        """
        compute angle between two points.
        returns angle in both radians and degrees
        """
        h = np.sqrt((pt2[0]-pt1[0])**2 + (pt2[1]-pt1[1])**2)
        o = pt2[1] - pt1[1]
        theta_r = np.asin(o/h)
        theta_d = np.rad2deg(theta_r)
        return theta_r, theta_d


if __name__ == "__main__":
    phwm = plot_hwm(var="zs")
    phwm.plot_scatter(model_runname="run6-5m-bldgs-3hr-tideloc4", domain="regular", fname="hwm.png")


    plt.show()


