import os
import pandas as pd
import geopandas as gpd
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib as mpl

class CompareForcingOutput():
    """docstring for xb_plotting_pt"""
    def __init__(self, var="H", xb_locs=[1], domain="large", tstart=0):
        self.model_runname = model_runname
        self.var = var
        self.xb_locs = xb_locs
        self.tstart = tstart

        self.file_dir = os.path.dirname(os.path.realpath(__file__))
        self.path_to_model = os.path.join(self.file_dir, "..", "..", "xbeach", "models", self.model_runname)
        self.path_to_forcing = os.path.join(self.file_dir, "..", "..", "data", "forcing",)

        self.read_domain(domain)
        self.read_forcing(xb_locs)

    def read_domain(self, domain):
        if domain == "small":
            fn = os.path.join(self.file_dir, "..", "..", "data", "xbeach-domain", "xbeach-domain-smaller-epsg32617.geojson")
        if domain == "medium":
            fn = os.path.join(self.file_dir, "..", "..", "data", "xbeach-domain", "xbeach-domain-epsg32617.geojson")
        if domain == "large":
            fn = os.path.join(self.file_dir, "..", "..", "data", "xbeach-domain", "xbeach-domain-larger-epsg32617.geojson")
        
        print(fn)
        self.domain = gpd.read_file(fn)

    def read_forcing(self, xb_locs):
        self.forcing_pt, self.fkeys = self.read_forcing_locs(xb_locs)
        
        if self.var=="H":
            fvar = "hs"
            ylabel = "Wave Height (m)"
        if self.var=="zs":
            fvar = "el"
            ylabel = "Water Elevation (m)"
        if self.var=="zs0":
            fvar = "el"
            ylabel = "Water Elevation - No Waves (m)"

        for cnt, xb_loc in enumerate(xb_locs):
            if xb_loc<5:
                frcng_dat = os.path.join(self.path_to_forcing, "xbeach{}-{}.dat" .format(xb_loc, self.fkeys[xb_loc]))
            else:
                frcng_dat = os.path.join(self.path_to_forcing, "xbeach{}.dat" .format(xb_loc))

            fn = os.path.join(self.file_dir, "..", "..", "..", "data", "forcing", frcng_dat)
            df_ = self.frcing_to_dataframe(fn)

            if cnt == 0:
                df = df_[["t_hr", fvar]].copy()
                df["{}-{}" .format(fvar, xb_loc)] = df[fvar]
                del df[fvar]
                continue

            df["{}-{}" .format(fvar, xb_loc)] = df_[fvar]
        self.forcing_df = df  

    def read_forcing_locs(self, xb_locs):
        fn = os.path.join(self.path_to_forcing, "forcing-pts.geojson")
        gdf = gpd.read_file(fn)
        gdf = gdf.to_crs(self.domain.crs)
        gdf = gdf.loc[gdf["id"].isin(xb_locs)]
        fkeys = {1: "sw", 2:"se", 3:"nw", 4:"ne"}
        return gdf, fkeys

    def compare_forcing2output(self, drawdomain=False, savefig=False):
        idx, idy, xgr, ygr, zgr = self.get_grid_indices()        
        if self.var=="H":
            fvar = "hs"
            ylabel = "Wave Height (m)"
        if self.var=="zs":
            fvar = "el"
            ylabel = "Water Elevation (m)"
        if self.var=="zs0":
            fvar = "el"
            ylabel = "Water Elevation - No Waves (m)"
        

        for cnt, xb_loc in enumerate(self.xb_locs):
            print(xb_loc)
            print(idx[cnt], idy[cnt])
            print("")
            fig, ax = plt.subplots(1,1)
            z, t = self.read_data_xarray_point(var=self.var, idx=idx[cnt], idy=idy[cnt], prnt_read=False)
            t_hr = t/3600 + self.tstart
            ax.plot(t_hr, z, color="dodgerblue", lw=2.5, label="XBeach")
            ax.plot(self.forcing_df["t_hr"], self.forcing_df["{}-{}" .format(fvar, xb_loc)], color="k", lw=1.5, label="ADCIRC/SWAN")
            ax.set_title("Save Point: {}" .format(xb_loc))
            ax.set_xlabel("Time (hrs)")
            ax.set_ylabel(ylabel)
            ax.legend()

            if savefig == True:
                fn = "frcng2output-{}.png" .format(self.xb_loc)
                plt.savefig(fn,
                            transparent=False, 
                            dpi=300,
                            bbox_inches='tight',
                            pad_inches=0.1,
                            )
                plt.close()


        if drawdomain:
            fig, ax = plt.subplots(1,1, figsize=(3,8))
            # self.domain.plot(ax=ax)
            # self.forcing_pt.plot(ax=ax, color="pink")

            # --- old
            # mask = (data_plot < -99999)
            # masked_array = np.ma.array(data_plot, mask=mask)
            cmap = mpl.cm.BrBG_r
            cmap.set_bad('bisque',1.)
            ax.pcolormesh(xgr, ygr, zgr, cmap=cmap)
            ax.set_xlabel("x (m)")
            ax.set_ylabel("y (m)")
            
            # cnt = 0
            # colors = sns.color_palette("husl")
            for cnt, xb_loc in enumerate(self.xb_locs):
                x, y = xgr[0,idx[cnt]], ygr[idy[cnt], 0]                
                ax.scatter(x, y, color="coral",s=50)
                ax.annotate("{}" .format(xb_loc), (x, y))

        if savefig == True:
            fn = "domain.png"
            plt.savefig(fn,
                        transparent=False, 
                        dpi=300,
                        bbox_inches='tight',
                        pad_inches=0.1,
                        )
            plt.close()

    def get_grid_indices(self):
        xgr, ygr, zgr = self.read_grid()
        (xo, yo), alfa_r = self.compute_grid_rotation()
        nx = np.shape(xgr)[1]
        ny = np.shape(xgr)[0]
        pt_x_wrld, pt_y_wrld = [], []
        idx, idy = [], []
        for x, pt_x in enumerate(xgr[0,:]):
            for y, pt_y in enumerate(ygr[:,0]):
                pt_x_wrld_ = xo + pt_x*np.cos(alfa_r) - pt_y*np.sin(alfa_r)
                pt_y_wrld_ = yo + pt_x*np.sin(alfa_r) + pt_y*np.cos(alfa_r)
                pt_x_wrld.append(pt_x_wrld_)
                pt_y_wrld.append(pt_y_wrld_)
                idx.append(x)
                idy.append(y)

        df = pd.DataFrame()
        df["pt_x_wrld"] = pt_x_wrld
        df["pt_y_wrld"] = pt_y_wrld
        df["idx"] = idx
        df["idy"] = idy
        df = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df["pt_x_wrld"], df["pt_y_wrld"]), crs=self.domain.crs)
        frcg_pts_gdf = gpd.sjoin_nearest(self.forcing_pt, df[["idx","idy", "geometry"]], how="left", distance_col="distance")
        
        idx = frcg_pts_gdf["idx"].values
        idy = frcg_pts_gdf["idy"].values

        return idx, idy, xgr, ygr, zgr



    def compute_grid_rotation(self):
        # getting exterior points 
        x = self.domain.geometry.exterior[0].xy[0]
        y = self.domain.geometry.exterior[0].xy[1]

        """ loop through exterior points; pull out grid width, length, origin, 
            and angle of rotation (theta).
        """
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
        
        pt1 = (xo,yo)
        pt2 = (xa,ya)
        h = np.sqrt((pt2[0]-pt1[0])**2 + (pt2[1]-pt1[1])**2)
        o = pt2[1] - pt1[1]
        theta_r = np.asin(o/h)
        theta_d = np.rad2deg(theta_r)
        return (xo, yo), theta_r



    def read_data_xarray_point(self, var, idx, idy, prnt_read=False, rtn_time_array=True):
        fn = os.path.join(self.path_to_model, "xboutput.nc")
        ds = xr.open_dataset(fn, chunks={"globaltime": 100})
        if prnt_read:
            print("Dataset object read:")
            print(ds)
            print("\n\n")
        
        # slice_data = ds[var].isel(globaltime=slice(t,t+1))
        slice_data = ds[var][:,idy,idx]

        if rtn_time_array:
            time = ds["globaltime"].values
            return slice_data.values, time
        else:
            return slice_data.values

if __name__ == "__main__":
    cfo = CompareForcingOutput(
            # model_runname="frun1-30m-bldgs-12hr-tideloc4", 
            var="zs0",    # zs, zs1, H
            xb_locs=[5],
            domain="medium",
            tstart=60)
    cfo.compare_forcing2output(drawdomain=True)



    plt.show()