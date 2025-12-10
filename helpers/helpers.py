import os
import numpy as np
import pandas as pd
import xarray as xr
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.ndimage as ndi
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling

class HelperFuncs():
    """
    Helper functions for plotting codes.
    Uses `paths.txt` to set a few variables. The following must be in paths.txt:
        • relative_path
        • path_to_model
        • path_to_forcing
    """
    def __init__(self):
        self.file_dir = os.path.dirname(os.path.realpath(__file__))
        self.project_dir = os.path.abspath(os.path.join(self.file_dir, ".."))
        self.read_paths()
        self.xboutput_filename = self.get_output_filename()
        self.make_directory(self.path_to_save_plot)

    def hello(self):
        print("hello world")

    def read_paths(self):
        """
        Reads paths from the file `paths.txt`.
        The paths specified in `paths.txt` are used in the plotting scripts.
        This function returns the paths as variables that are then accessible within 
          this helper function as instance variables.
        For example, if paths.txt consists of:
            path_to_model = "../models"
            path_to_buildings = "../buildings"
        then the following varialbes will be created: self.path_to_model, and
          self.path_to_buildings. 
        The keyword "relative_path" in `paths.txt` sets whether the current
          file path of this helpers.py file is appended to the path provided. 
        """
        # TODO: add check for required variables
        relative_path = False
        fn = os.path.join(self.project_dir, "paths.txt")
        with open(fn,'r') as f:
            for cnt, line in enumerate(f.readlines()):
                line = line.strip()
                if "relative_path" in line:
                    exec(line, globals())
                    if line.split("=")[1].strip() == "False":
                        relative_path = False
                    elif line.split("=")[1].strip() == "True":
                        relative_path = True
                if line and '=' in line:
                    key, value = line.split('=', 1)
                    key = key.strip()
                    value = value.strip()
                    value = value.split("/")
                    if relative_path==True:
                        pth = os.path.join(self.project_dir, *value)                
                    else:
                        value.insert(0, "/")
                        pth = os.path.join(*value)
                    setattr(self, key, pth)
                    print("  successfully set {}" .format(key))
        self.model_runname = self.path_to_model.split(os.sep)[-1]
        print("paths set for {}" .format(self.model_runname))

    def get_output_filename(self, model_dir=None):
        """
        Returns the output file name that is located in the `path_to_model` 
          directory.
        """
        if model_dir == None:
            model_dir = self.path_to_model
        files = os.listdir(model_dir)        
        nc_files = [i for i in files if ".nc" in i]
        if len(nc_files) > 0:
            # fn  = [i for i in files if ".nc" in i][0]
            fn = nc_files[0]

        else:
            fn = None
        return fn

    def get_figsize(self, domain_size):
        """
        Returns figure size based on the domain being considered. 
        """
        if domain_size=="micro":
            figsize=(7,5)
        else:
            figsize=(3,8)
        return figsize

    def read_max_xarray(self, var):
        """
        Reads the maximum value from the xarray dataset.
        Inputs:
            var: variable to read, e.g., zs1
        Returns: 
            data: 2D numpy array.
        """
        fn = os.path.join(self.path_to_model, self.xboutput_filename)
        ds = xr.open_dataset(fn, chunks={"globaltime": 100})
        
        max_vals = ds[var].max(dim="globaltime").values[:,:]
        return max_vals
    
    def read_dims_xarray(self):
        """
        reads the dimensions of the xbeach output:
        """
        fn = os.path.join(self.path_to_model, self.xboutput_filename)
        ds = xr.open_dataset(fn, chunks={"globaltime": 100})
        
        nx = ds.sizes["nx"]
        ny = ds.sizes["ny"]
        return (ny, nx)
    
    def get_resolution(self):
        fn_params = os.path.join(self.path_to_model, "params.txt")
        with open(fn_params,'r') as f:
            for cnt, line in enumerate(f.readlines()):
                if "dx" in line:
                    if "vardx" in line:
                        continue
                    l_ = [i.strip() for i in line.split()]
                    dx = float(l_[-1])
                if "dy" in line:
                    l_ = [i.strip() for i in line.split()]
                    dy = float(l_[-1])
        return dx, dy

    def get_origin(self, model_dir=None):
        if model_dir == None:
            model_dir = self.path_to_model
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
        return xo, yo, theta
    
    def read_from_params(self, fn_params=None, var=None):
        if fn_params == None:
            fn_params = os.path.join(self.path_to_model, "params.txt")
        with open(fn_params,'r') as f:
            for cnt, line in enumerate(f.readlines()):
                if var in line:
                    l_ = [i.strip() for i in line.split()]
                    val = float(l_[-1])
                    break
        return val

    def read_transect_data_xarray(self, var, idy, t_idx):
        """
        reads a transect from the xarray output dataset
        """
        fn = os.path.join(self.path_to_model, self.xboutput_filename)
        ds = xr.open_dataset(fn, chunks={"globaltime": 100})
        slice_data = ds[var][t_idx,idy,:]
        return slice_data.values

    def read_3d_data_xarray(self, var):
        """
        reads the full 3D array (x,y, time) from the netcdf output file
        Inputs:
            var: variable to read, e.g., zs1
        Returns: 
            data: 2D numpy array.
        """
        fn = os.path.join(self.path_to_model, self.xboutput_filename)
        ds = xr.open_dataset(fn, chunks={"globaltime": 100})
        return ds[var][:,:,:].values

    def read_3d_data_xarray_nonmem(self, var):
        """
        reads the full 3D array (x,y, time) from the netcdf output file
        Inputs:
            var: variable to read, e.g., zs1
        Returns: 
            data: 2D numpy array.
        """
        fn = os.path.join(self.path_to_model, self.xboutput_filename)
        ds = xr.open_dataset(fn, chunks={"globaltime": -1, "x": -1, "y": 400})
        return ds[var][:,:,:]

    def read_2d_data_xarray_timestep(self, var, t, model_dir=None):        
        """
        Reads xarray data for entire domain at specified time step.
        Inputs:
            var: variable to read, e.g., zs1
            t: time step to read
        Returns: 
            data: 2D numpy array.
        """
        if model_dir == None:
            model_dir = self.path_to_model
        xb_output = self.get_output_filename(model_dir)
        fn = os.path.join(model_dir, xb_output)
        ds = xr.open_dataset(fn, chunks={"globaltime": 100})

        slice_data = ds[var].isel(globaltime=slice(t,t+1))
        return slice_data.values[0,:,:]

    def read_pt_data_xarray(self, var, idx, idy):
        """
        Reads xarray data at a specific point. 
        Inputs: 
            var: variable to read, e.g., zs1
            idx: index in x-direction.
            idy: index in y-direction.
        Returns:
            time series of data at the point idx, idy. 
        """
        fn = os.path.join(self.path_to_model, self.xboutput_filename)
        ds = xr.open_dataset(fn, chunks={"globaltime": 100})        
        slice_data = ds[var][:,idy,idx]
        return slice_data.values

    def read_npy(self, stat, run=None):
        """
        TODO: add docstring
        """
        if run == None:
            fn = os.path.join(self.path_to_save_plot, "{}.npy" .format(stat))
        else:
            fn = os.path.join(self.path_to_save_plot, "..", run,  "{}.npy" .format(stat))
        rmax = np.load(fn)
        return rmax

    def read_time_xarray(self):
        """
        Reads time from the xbeach output.
        Returns:
            time: an array representing time steps
        """
        fn = os.path.join(self.path_to_model, self.xboutput_filename)
        ds = xr.open_dataset(fn, chunks={"globaltime": 100})
        time = ds["globaltime"].values
        return time

    def reproject_raster(self, path_to_dem, epsg):
        """
        function to reproject raster from current epsg to local utm epsg.
        Note that a temporary tiff file is written to "temp.tiff". This file is
        later removed when it is no longer needed.
        """
        with rasterio.open(path_to_dem) as src:
            transform, width, height = rasterio.warp.calculate_default_transform(
                src.crs, epsg, src.width, src.height, *src.bounds)
            kwargs = src.meta.copy()
            kwargs.update({
                "crs": epsg,
                "transform": transform,
                "width": width,
                "height": height
            })

            with rasterio.open("temp.tiff", "w", **kwargs) as dst:
                for i in range(1, src.count + 1):
                    d = reproject(
                        source=rasterio.band(src, i),
                        destination=rasterio.band(dst, i),
                        src_transform=src.transform,
                        src_crs=src.crs,
                        dst_transform=transform,
                        dst_crs=epsg,
                        resampling=Resampling.nearest)


    def read_buildings(self, run_w_bldgs=None, model_dir=None):
        """
        TODO: add docstring
        """
        if model_dir == None:
            model_dir = self.path_to_model
        if run_w_bldgs == None:
            fn_zgrid = os.path.join(model_dir, "z.grd")
        else:
            fn_zgrid = os.path.join(model_dir, "..", run_w_bldgs, "z.grd")
        zs = []
        with open(fn_zgrid,'r') as f:
            for cnt, line in enumerate(f.readlines()):
                z_ = [float(i.strip()) for i in line.split()]
                zs.append(z_)
        zgr = np.array(zs)
        mask = (zgr != 10)
        bldgs = np.ma.array(zgr, mask=mask)
        return bldgs

    def frcing_to_dataframe(self, n_header=3, n_var=7):
        """
        TODO: add docstring
        """
        t, el, wx, wy, hs, Tp, wavedir = [], [], [], [], [], [], [],
        with open(self.path_to_forcing,'r') as f:
            for cnt, line in enumerate(f.readlines()):
                if cnt < n_header:
                    if "VARIABLES" in line:
                        var = [x.strip() for x in line.split()]
                        var = [i for i in var if i!="VARIABLES"]
                        var = [i for i in var if i!="="]
                    continue
                t_, el_, wx_, wy_, hs_, Tp_, wavedir_ = [float(x.strip()) for x in line.split()]
                t.append(t_)
                el.append(el_)
                wx.append(wx_)
                wy.append(wy_)
                hs.append(hs_)
                Tp.append(Tp_)

                wavedir_ = self.cartesian_to_nautical_angle(wavedir_)
                wavedir_ = self.nautical_to_xbeach_angle(wavedir_, alfa=55.92839019260679)

                wavedir.append(wavedir_)

        # TODO confirm unit conversions with Don
        df = pd.DataFrame()
        df["t"] = t
        df["el"] = el
        df["wx"] = wx
        df["wy"] = wy
        df["hs"] = hs
        df["Tp"] = Tp
        df["wavedir"] = wavedir

        df["el"] = df["el"]*0.3048
        df["hs"] = df["hs"]*0.3048

        dt = (df["t"].iloc[1] - df["t"].iloc[0])*60*60         # tiime setp in seconds; converting from hours.
        df["t_sec"] = np.linspace(0, (len(df)-1)*dt, len(df))
        df["t_hr"] = df["t_sec"]/3600
        return df

    def read_grid(self, model_dir=None):
        """
        TODO: add docstring
        """
        # -- reading xgrid
        if model_dir == None:
            model_dir = self.path_to_model
        xgrid = os.path.join(model_dir, "x.grd")
        with open(xgrid,'r') as f:
            for cnt, line in enumerate(f.readlines()):
                xs = [float(i.strip()) for i in line.split()]
                if cnt == 0:
                    break
        
        # -- reading ygrid
        ys = []
        ygrid = os.path.join(model_dir, "y.grd")
        with open(ygrid,'r') as f:
            for cnt, line in enumerate(f.readlines()):
                y_ = [float(i.strip()) for i in line.split()][0]
                ys.append(y_)
        
        # -- reading zgrid
        zgr = np.zeros((len(ys), len(xs)))
        zgrid = os.path.join(model_dir, "z.grd")
        with open(zgrid,'r') as f:
            for cnt, line in enumerate(f.readlines()):
                z_ = [float(i.strip()) for i in line.split()]
                zgr[cnt,:] = z_
        
        # -- creating xgr, ygr mesh
        xgr, ygr = np.meshgrid(xs, ys)
        return xgr, ygr, zgr

    def xy_to_grid_index(self, xgr, ygr, xy):
        """
        Returns the indicies in xgr, ygr that are nearest to the xy points. 
        Useful for when the grid is not a 1 m resolution. 
        For example, with 2 m resolution grid, and want xy points at (200, 100)
        this function would return (100, 50).
        Inputs: 
            xgr: xgrid
            ygr: ygrid
            xy: tuple of x-y locations

        Returns: 
            (idx, idy): tuple of index for x and index for y
        """
        idx = np.argmin(np.abs(xgr[0,:] - xy[0]))
        idy = np.argmin(np.abs(ygr[:,0] - xy[1]))        
        return (idx,idy)

    def tstartstop_to_tindex(self, tstart, tstop, time):
        """ 
        returns the indicies nearest to the provided start and stop times
        """
        tstart_idx = np.argmin(np.abs(time-tstart*3600))
        tstop_idx  = np.argmin(np.abs(time-tstop*3600))
        return tstart_idx, tstop_idx
    
    def time_to_tindex(self, time_wanted, time):
        t_idx = np.argmin(np.abs(time-time_wanted))
        return t_idx.item()


    def save_fig(self, fig, fn=None, **kwargs):
        """
        Saves figures
        Inputs:
            fig: matplotlib figure. 
            fn: filename; if None, no figure is created
            **kwargs: keyword args to pass to matplotlib savefig function. 
        """
        if fn != None:
            fn = os.path.join(self.path_to_save_plot, fn)
            plt.savefig(fn,
                        pad_inches=0.1,
                        bbox_inches='tight',
                        **kwargs,
                        # transparent=True, 
                        # dpi=self.dpi,
                        )
            plt.close()

    def get_H(self, z, detrend=True):
        if detrend:
            z = z - np.mean(z)  # de-trend signal with mean
        
        # The sign of the (detrended) elevation at each point
        signs = np.sign(z)
        # Find where the sign changes.
        zero_crossing_indices = np.where(np.diff(signs) != 0)[0]

        up_crossings = zero_crossing_indices[np.where(signs[zero_crossing_indices] < signs[zero_crossing_indices + 1])[0]]
        # Ensure we have pairs of up-crossings to define full waves
        start_indices = up_crossings[:-1]
        end_indices = up_crossings[1:]

        # Use a list comprehension to get max and min values for each segment
        crests = [np.max(z[start:end]) for start, end in zip(start_indices, end_indices)]
        troughs = [np.min(z[start:end]) for start, end in zip(start_indices, end_indices)]

        # Convert to NumPy arrays for vectorized subtraction
        wave_heights = np.array(crests) - np.array(troughs)
        return wave_heights


    def get_T(self, z, t, detrend=True):
        if detrend:
            z = z - np.mean(z)  # de-trend signal with mean
        
        # The sign of the (detrended) elevation at each point
        signs = np.sign(z)
        # Find where the sign changes.
        zero_crossing_indices = np.where(np.diff(signs) != 0)[0]

        up_crossings = zero_crossing_indices[np.where(signs[zero_crossing_indices] < signs[zero_crossing_indices + 1])[0]]
        t_ = t[up_crossings] # get time of upcrossings
        T = np.diff(t_) # now get difference from each upcrossing time
        return T

    def assign_max_to_bldgs(self, data, bldgs):
        max_H = np.empty(np.shape(data))
        max_H[:] = np.nan
        mask = np.ma.getmask(bldgs)
        labeled_mask, num_features = ndi.label(~mask)
        for i in range(num_features+1):
            if i == 0:
                continue
            m_ = labeled_mask==i
            m_ = ~m_
            m_ = np.pad(m_, pad_width=1, mode="constant", constant_values=True)

            shifted_up = m_[2:, 1:-1]
            shifted_down = m_[:-2, 1:-1]
            shifted_left = m_[1:-1, 2:]
            shifted_right = m_[1:-1, :-2]
            original_mask_trimmed = m_[1:-1, 1:-1]

            offset_mask = original_mask_trimmed & shifted_up & shifted_down & shifted_left & shifted_right
            offset_mask = ~offset_mask

            max_H[labeled_mask==i] = np.nanmax(data[offset_mask])

        return max_H
        
    def compute_Hs(self, H):
        if len(H) == 0:
            return 0
        H_one_third = np.quantile(H, q=2/3)
        H = H[H>H_one_third]
        if len(H) == 0:
            return 0

        Hs = np.mean(H)

        return Hs.item()

    def cartesian_to_nautical_angle(self, deg):
        """ converting from cartesian to nautical angles for xbeach input
        Cartesian: waves traveling TO east are zero and counterclockwise is positive.
        Nautical: waves traveling FROM North are zero and clockwise is positive. 
        """
        if (deg>=0) & (deg <= 270):
           return (270-deg)
        elif (deg>270) & (deg<360):
           return (270-deg)+360
        else:
            raise ValueError("{} must be between 0 and 360." .format(deg))

    def nautical_to_xbeach_angle(self, deg, alfa):
        """
        """
        deg = deg + alfa 
        if deg > 360:
            deg -= 360
        elif deg < 0:
            deg += 360
        return deg

    def make_directory(self, path_out):
        """
        make directory if it doesn't exist
        """
        if not os.path.exists(path_out):
            os.makedirs(path_out, exist_ok=True)
        return path_out

    def var2label(self, var):
        v2l = { "el":"Water Elevation",
                "hs": "Sig. Wave Height",
                "Tp": "Peak Period",
                "wavedir": "Wave Direction",
                "zs": "Water Elevation",
                "zs0": "Surge Level",
                "zs1": "Water Elevation Above Surge",
        }
        v2y = { "el": "Water Elevation (m)",
                "hs": "Sig. Wave Height (m)",
                "Tp": "Peak Period (s)",
                "wavedir": "Wave Direction",
                "zs": "Water Elevation (m)",
                "zs0": "Surge Level (m)",
                "zs1": "Water Elevation Above Surge (m)",
        }
        
        c = {"el": 0, "hs": 1, "Tp": 2, "wavedir": 3, "zs": 4, "zs0": 5, "zs1": 6}
        colors = sns.color_palette("crest", n_colors=len(c.keys()))
        color = colors[c[var]]

        return v2l[var], v2y[var], color
    

    def xbeach_duration_to_start_stop(self, duration):
        duration_to_start_stop = {
                    0.5: {"start": 66.25, "stop":  66.75},
                    1:   {"start": 66,    "stop":  67},
                    2:   {"start": 65.25, "stop":  67.25},
                    3:   {"start": 65,    "stop":  68},
                    4:   {"start": 64,    "stop":  68},
                    6:   {"start": 63,    "stop":  69},
                    8:   {"start": 62,    "stop":  70},
                    10:  {"start": 61,    "stop":  71},
                    12:  {"start": 60,    "stop":  72},
                    16:  {"start": 58,    "stop":  74}
                                }
        t_start = duration_to_start_stop[duration]["start"]
        t_stop = duration_to_start_stop[duration]["stop"]
        return t_start, t_stop

    def remove_frame(self, ax):
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.get_xaxis().set_ticks([])
        ax.get_yaxis().set_ticks([])

    def check_domain_size_wave_stat(self, run1_max, run2_max):
        """
        Checks that the domain size of the two runs being compared is identical.
        If they are not the same, then resize one to match the other. 
        This is used when comparing two runs at different resolutions. 
        """
        run1_shape = np.shape(run1_max)
        run2_shape = np.shape(run2_max)

        if run1_shape != run2_shape:
            r1_to_r2 = np.divide(run1_shape,run2_shape)
            r2_to_r1 = np.divide(run2_shape,run1_shape)
            scale = np.maximum(r1_to_r2, r2_to_r1)

            # if scale[0]!= scale[1]:
            #     raise ValueError("need domains to be same proportion.")

            if (scale == r1_to_r2).all():
                temp = np.repeat(run2_max, scale[0], axis=0)
                run2_max = np.repeat(temp, scale[0], axis=1)
            elif (scale == r2_to_r1).all():
                temp = np.repeat(run1_max, scale[0], axis=0)
                run1_max = np.repeat(temp, scale[0], axis=1)

        return run1_max, run2_max


    def rmse(self, predictions, targets):
        return np.sqrt(((predictions - targets) ** 2).mean())
    def mae(self, predictions, targets):
        return np.mean(np.abs(predictions - targets))

if __name__ == '__main__':
    hf = HelperFuncs()


















