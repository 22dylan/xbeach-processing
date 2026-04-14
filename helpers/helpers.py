import os
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Union, Any

import numpy as np
import pandas as pd
import geopandas as gpd
import xarray as xr
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.ndimage as ndi
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling

class HelperFuncs():
    """
    Helper functions for plotting and processing XBeach model output.
    Uses `paths.txt` to set model and data paths.
    """
    def __init__(self):
        self.file_path = Path(__file__).resolve()
        self.project_dir = self.file_path.parent.parent
        self.read_paths()
        self.xboutput_filename = self.get_output_filename()
        
        if hasattr(self, 'path_to_save_plot'):
            self.make_directory(self.path_to_save_plot)

    def read_paths(self):
        """
        Reads paths from the file `paths.txt`.
        The paths specified in `paths.txt` are used in the plotting scripts.
        Each line in paths.txt should be in the format: key = value
        """
        paths_fn = self.project_dir / "paths.txt"
        if not paths_fn.exists():
            print(f"Warning: {paths_fn} not found.")
            return

        config = {}
        relative_path = False
        
        with open(paths_fn, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#') or '=' not in line:
                    continue
                
                key, value = line.split('=', 1)
                key = key.strip()
                value = value.strip()
                
                if key == "relative_path":
                    relative_path = value.lower() == "true"
                    continue
                
                config[key] = value

        for key, value in config.items():
            if relative_path:
                pth = self.project_dir / value
            else:
                # Handle absolute paths or paths starting from root
                if value.startswith('/'):
                    pth = Path(value)
                else:
                    pth = Path('/') / value
            
            setattr(self, key, str(pth))
            print(f"  successfully set {key}")

        if hasattr(self, 'path_to_model'):
            self.model_runname = Path(self.path_to_model).name
            self.check_hotstart()
            print(f"paths set for {self.model_runname}")
        else:
            print("Warning: path_to_model not set in paths.txt")

    def check_hotstart(self) -> None:
        """
        Checks if the model run is a hotstart run by looking for params.txt 
        in the model directory.
        """
        params_fn = Path(self.path_to_model) / "params.txt"
        if params_fn.exists():
            self.hotstart_run = False
        else:
            self.hotstart_run = True
            self.hotstart_runs = self.set_hotstart_runs()

    def get_first_model_dir(self) -> str:
        """
        Returns the path to the first model directory (useful for hotstart runs).
        """
        if self.hotstart_run:
            if not hasattr(self, 'hotstart_runs') or not self.hotstart_runs:
                return self.path_to_model
            return str(Path(self.path_to_model) / self.hotstart_runs[0])
        else:
            return self.path_to_model

    def get_output_filename(self) -> Optional[str]:
        """
        Returns the output file name (.nc) that is located in the model directory.
        """
        model_dir = Path(self.get_first_model_dir())
        if not model_dir.exists():
            return None
        
        nc_files = list(model_dir.glob("*.nc"))
        if nc_files:
            return nc_files[0].name
        return None

    def get_dataset(self, chunks: Optional[Dict[str, int]] = None) -> xr.Dataset:
        """
        Returns an xarray dataset for the model run, handling hotstarts automatically.
        """
        if chunks is None:
            chunks = {"globaltime": 100}
            
        if self.hotstart_run:
            return self.get_hotstart_ds(chunks=chunks)
        else:
            fn = Path(self.path_to_model) / self.xboutput_filename
            if not fn.exists():
                raise FileNotFoundError(f"Output file {fn} not found.")
            return xr.open_dataset(fn, chunks=chunks)

    def get_figsize(self, domain_size: str) -> Tuple[int, int]:
        """
        Returns figure size based on the domain being considered. 
        """
        if domain_size == "micro":
            return (7, 5)
        else:
            return (3, 8)

    def get_hotstart_ds(self, chunks: Optional[Dict[str, int]] = None) -> xr.Dataset:
        """
        Opens multiple hotstart NetCDF files and corrects the resetting 'globaltime'.
        """
        if chunks is None:
            chunks = {"globaltime": 100}
            
        datasets = []
        cumulative_time = 0.0
        
        for run in self.hotstart_runs:
            fn = Path(self.path_to_model) / run / self.xboutput_filename
            if not fn.exists():
                print(f"Warning: {fn} not found, skipping.")
                continue
                
            ds = xr.open_dataset(fn, chunks=chunks)
            
            # Update globaltime: add the cumulative_time from previous runs
            ds["globaltime"] = ds["globaltime"] + cumulative_time
            datasets.append(ds)
            
            # Update cumulative_time using the last time step plus dt
            if len(ds["globaltime"]) > 1:
                dt = ds["globaltime"][1].values - ds["globaltime"][0].values
            else:
                # Fall back to params if only one time step
                params_fn = Path(self.path_to_model) / run / "params.txt"
                dt = float(self.read_from_params(fn_params=str(params_fn), var="tintg"))
            
            cumulative_time = float(ds["globaltime"][-1].values) + dt
            
        if not datasets:
            raise FileNotFoundError(f"No hotstart datasets found in {self.path_to_model}")
            
        return xr.concat(datasets, dim="globaltime", data_vars="all")

    def read_max_xarray(self, var: str) -> np.ndarray:
        """
        Reads the maximum value from the xarray dataset.
        Inputs:
            var: variable to read, e.g., zs1
        Returns: 
            data: 2D numpy array.
        """
        ds = self.get_dataset()
        return ds[var].max(dim="globaltime").values

    def read_dims_xarray(self) -> Tuple[int, int]:
        """
        Reads the dimensions of the xbeach output.
        Returns:
            (ny, nx): tuple of dimensions
        """
        ds = self.get_dataset()
        return (ds.sizes["ny"], ds.sizes["nx"])

    def get_resolution(self) -> Tuple[float, float]:
        """
        Reads dx and dy from params.txt.
        """
        model_dir = Path(self.get_first_model_dir())
        params_fn = model_dir / "params.txt"
        
        dx, dy = 0.0, 0.0
        with open(params_fn, 'r') as f:
            for line in f:
                if "dx" in line and "vardx" not in line:
                    dx = float(line.split()[-1])
                if "dy" in line:
                    dy = float(line.split()[-1])
        return dx, dy

    def get_origin(self) -> Tuple[float, float, float]:
        """
        Reads xo, yo, and theta from params.txt.
        """
        model_dir = Path(self.get_first_model_dir())
        params_fn = model_dir / "params.txt"
        
        xo, yo, theta = 0.0, 0.0, 0.0
        with open(params_fn, 'r') as f:
            for line in f:
                parts = line.split()
                if not parts: continue
                if "xo" in line:
                    xo = float(parts[-1])
                if "yo" in line:
                    yo = float(parts[-1])
                if "theta" in line:
                    theta = float(parts[-1])
        return xo, yo, theta
    
    def read_from_params(self, fn_params: Optional[str] = None, var: Optional[str] = None) -> Any:
        """
        Reads a specific variable from params.txt.
        """
        if fn_params is None:
            fn_params = str(Path(self.path_to_model) / "params.txt")
        
        if var is None:
            return None

        val = None
        with open(fn_params, 'r') as f:
            for line in f:
                if var in line:
                    parts = line.split()
                    try:
                        val = float(parts[-1])
                    except ValueError:
                        val = parts[-1]
                    break
        return val

    def read_transect_data_xarray(self, var: str, idy: int, t_idx: int) -> np.ndarray:
        """
        Reads a transect from the xarray output dataset.
        """
        ds = self.get_dataset()
        return ds[var][t_idx, idy, :].values

    def read_3d_data_xarray(self, var: str) -> np.ndarray:
        """
        Reads the full 3D array (time, y, x) from the netcdf output file.
        """
        ds = self.get_dataset()
        return ds[var].values

    def read_3d_data_xarray_nonmem(self, var: str) -> xr.DataArray:
        """
        Reads the full 3D DataArray without loading all into memory.
        """
        ds = self.get_dataset(chunks={"globaltime": -1, "nx": -1, "ny": 400})
        return ds[var]

    def read_2d_data_xarray_timestep(self, var: str, t: int, hsrun: Optional[str] = None) -> np.ndarray:        
        """
        Reads xarray data for entire domain at specified time step.
        """
        if hsrun is None:
            model_dir = Path(self.get_first_model_dir())
        else:
            model_dir = Path(self.path_to_model) / hsrun
            
        fn = model_dir / self.xboutput_filename
        ds = xr.open_dataset(fn, chunks={"globaltime": 100})
        return ds[var].isel(globaltime=t).values

    def read_pt_data_xarray(self, var: str, idx: int, idy: int) -> np.ndarray:
        """
        Reads xarray data at a specific point. 
        """
        ds = self.get_dataset()
        return ds[var][:, idy, idx].values

    def read_npy(self, stat: str, run: Optional[str] = None) -> np.ndarray:
        """
        Reads .npy files from the results directory.
        """
        if run is None:
            fn = Path(self.path_to_save_plot) / f"{stat}.npy"
        else:
            fn = Path(self.path_to_save_plot).parent / run / f"{stat}.npy"
        return np.load(fn)

    def read_dat(self, stat: str) -> np.ndarray:
        """
        Reads .dat files from the results directory.
        """
        fn = Path(self.path_to_save_plot) / f"{stat}.dat"
        return np.loadtxt(fn)

    def read_time_xarray(self) -> np.ndarray:
        """
        Reads time from the xbeach output.
        """
        ds = self.get_dataset()
        return ds["globaltime"].values

    def reproject_raster(self, path_to_dem: str, epsg: Union[int, str]) -> None:
        """
        Function to reproject raster from current epsg to local utm epsg.
        Note that a temporary tiff file is written to "temp.tiff".
        """
        with rasterio.open(path_to_dem) as src:
            transform, width, height = calculate_default_transform(
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
                    reproject(
                        source=rasterio.band(src, i),
                        destination=rasterio.band(dst, i),
                        src_transform=src.transform,
                        src_crs=src.crs,
                        dst_transform=transform,
                        dst_crs=epsg,
                        resampling=Resampling.nearest)

    def read_removed_bldgs(self) -> Optional[np.ndarray]:
        """
        Reads removed buildings mask from various possible files.
        """
        model_dir = Path(self.path_to_model)
        possible_files = [
            "removed_bldgs.npy",
            "stat_removed_bldgs.dat",
            "removed_bldgs.dat"
        ]
        
        for fn in possible_files:
            file_path = model_dir / fn
            if file_path.exists():
                if file_path.suffix == ".npy":
                    data = np.load(file_path)
                else:
                    data = np.loadtxt(file_path)
                return data.astype(bool)
        return None

    def read_buildings(self, run_w_bldgs: Optional[str] = None, hsrun: Optional[str] = None) -> np.ma.MaskedArray:
        """
        Reads building grid (z.grd) and returns a masked array where buildings are present.
        """
        if hsrun is None:
            model_dir = Path(self.get_first_model_dir())
        else:
            model_dir = Path(self.path_to_model) / hsrun
            
        if run_w_bldgs is None:
            fn_zgrid = model_dir / "z.grd"
        else:
            fn_zgrid = Path(self.path_to_model) / run_w_bldgs / "z.grd"
            
        if not fn_zgrid.exists():
            raise FileNotFoundError(f"Building grid file {fn_zgrid} not found.")

        zgr = np.loadtxt(fn_zgrid)
        # Buildings are usually represented by values >= 10 in z.grd
        mask = (zgr < 10)
        return np.ma.array(zgr, mask=mask)
        
    def frcing_to_dataframe(self, n_header: int = 3) -> pd.DataFrame:
        """
        Reads XBeach forcing file and returns a pandas DataFrame.
        """
        forcing_path = Path(self.path_to_forcing)
        if not forcing_path.exists():
            raise FileNotFoundError(f"Forcing file {forcing_path} not found.")

        # Try to parse variables from header
        var_names = []
        with open(forcing_path, 'r') as f:
            for i in range(n_header):
                line = f.readline()
                if "VARIABLES" in line:
                    var_names = [v.strip() for v in line.split('=')[-1].split()]

        # If var_names not found in header, use defaults
        if not var_names:
            var_names = ["t", "el", "wx", "wy", "hs", "Tp", "wavedir"]

        # Read data
        data = np.loadtxt(forcing_path, skiprows=n_header)
        
        # If number of columns doesn't match var_names, adjust
        if data.shape[1] > len(var_names):
            # Probably has vx, vy too
            extra_vars = [f"var_{i}" for i in range(len(var_names), data.shape[1])]
            var_names.extend(extra_vars)
        elif data.shape[1] < len(var_names):
            var_names = var_names[:data.shape[1]]

        df = pd.DataFrame(data, columns=var_names)

        # Apply angle conversions to wavedir
        if "wavedir" in df.columns:
            df["wavedir"] = df["wavedir"].apply(self.cartesian_to_nautical_angle)
            # Hardcoded alfa for now, as in original code
            alfa = 55.92839019260679
            df["wavedir"] = df["wavedir"].apply(lambda x: self.nautical_to_xbeach_angle(x, alfa))

        # Unit conversions (ft to m) - TODO: verify if this is always needed
        if "el" in df.columns:
            df["el"] *= 0.3048
        if "hs" in df.columns:
            df["hs"] *= 0.3048

        # Add time in seconds and hours
        if "t" in df.columns:
            # Assuming 't' is in hours based on original code's dt calculation
            dt_sec = (df["t"].iloc[1] - df["t"].iloc[0]) * 3600 if len(df) > 1 else 0
            df["t_sec"] = np.arange(len(df)) * dt_sec
            df["t_hr"] = df["t_sec"] / 3600
            
        return df

    def read_grid(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Reads the x, y, and z grids.
        """
        model_dir = Path(self.get_first_model_dir())
        
        x_fn = model_dir / "x.grd"
        y_fn = model_dir / "y.grd"
        z_fn = model_dir / "z.grd"
        
        if not all(p.exists() for p in [x_fn, y_fn, z_fn]):
            raise FileNotFoundError(f"One or more grid files missing in {model_dir}")

        # x.grd and y.grd can be 1D or 2D. XBeach often uses 1D files for regular grids.
        # Let's try to load them and handle the meshgrid creation.
        
        x_raw = np.loadtxt(x_fn)
        y_raw = np.loadtxt(y_fn)
        zgr = np.loadtxt(z_fn)
        
        if x_raw.ndim == 1 and y_raw.ndim == 1:
            xgr, ygr = np.meshgrid(x_raw, y_raw)
        else:
            xgr, ygr = x_raw, y_raw
            
        return xgr, ygr, zgr

    def read_coast(self) -> gpd.GeoDataFrame:
        """
        Reads coastline geodataframe.
        """
        return gpd.read_file(self.path_to_coast)

    def read_bldgs_geodataframe(self) -> gpd.GeoDataFrame:
        """
        Reads building geodataframe and sets VDA_id as index.
        """
        gdf = gpd.read_file(self.path_to_bldgs)
        if "VDA_id" in gdf.columns:
            gdf.set_index("VDA_id", inplace=True)
        return gdf

    def xy_to_grid_index(self, xgr: np.ndarray, ygr: np.ndarray, xy: Tuple[float, float]) -> Tuple[int, int]:
        """
        Returns the indices in xgr, ygr that are nearest to the xy points. 
        Useful for when the grid is not at 1 m resolution. 
        """
        idx = np.argmin(np.abs(xgr[0, :] - xy[0]))
        idy = np.argmin(np.abs(ygr[:, 0] - xy[1]))        
        return (int(idx), int(idy))

    def tstartstop_to_tindex(self, tstart: float, tstop: float, time: np.ndarray) -> Tuple[int, int]:
        """ 
        Returns the indices nearest to the provided start and stop times (in hours).
        """
        tstart_idx = np.argmin(np.abs(time - tstart * 3600))
        tstop_idx  = np.argmin(np.abs(time - tstop * 3600))
        return (int(tstart_idx), int(tstop_idx))
    
    def time_to_tindex(self, time_wanted: float, time: np.ndarray) -> int:
        """
        Returns the index nearest to the provided time (in seconds).
        """
        t_idx = np.argmin(np.abs(time - time_wanted))
        return int(t_idx)

    def save_fig(self, fig: plt.Figure, fn: Optional[str] = None, **kwargs: Any) -> None:
        """
        Saves figures to the results directory.
        """
        if fn is not None:
            save_path = Path(self.path_to_save_plot) / fn
            fig.savefig(save_path,
                        pad_inches=0.1,
                        bbox_inches='tight',
                        **kwargs)
            plt.close(fig)

    def get_H(self, z: np.ndarray, detrend: bool = True) -> np.ndarray:
        """
        Returns an array of wave heights from a time series of water elevations.
        Wave height defined as peak to trough; must cross through zero.
        """
        if detrend:
            z = z - np.mean(z)
        
        signs = np.sign(z)
        zero_crossing_indices = np.where(np.diff(signs) != 0)[0]
        up_crossings = zero_crossing_indices[np.where(signs[zero_crossing_indices] < signs[zero_crossing_indices + 1])[0]]
        
        if len(up_crossings) < 2:
            return np.array([])

        start_indices = up_crossings[:-1]
        end_indices = up_crossings[1:]

        crests = [np.max(z[start:end]) for start, end in zip(start_indices, end_indices)]
        troughs = [np.min(z[start:end]) for start, end in zip(start_indices, end_indices)]

        return np.array(crests) - np.array(troughs)

    def get_T(self, z: np.ndarray, t: np.ndarray, detrend: bool = True) -> np.ndarray:
        """
        Returns an array of periods from a time series of water elevations.
        Period defined as time between up-crossings.
        """
        if detrend:
            z = z - np.mean(z)
        
        signs = np.sign(z)
        zero_crossing_indices = np.where(np.diff(signs) != 0)[0]
        up_crossings = zero_crossing_indices[np.where(signs[zero_crossing_indices] < signs[zero_crossing_indices + 1])[0]]
        
        if len(up_crossings) < 2:
            return np.array([])
            
        t_ = t[up_crossings]
        return np.diff(t_)

    def assign_max_to_bldgs(self, data: np.ndarray, bldgs: np.ma.MaskedArray) -> np.ndarray:
        """
        Assigns maximum value to buildings;
        considers one cell to left, right, above, and below each building.
        """
        max_bldg = np.full(data.shape, np.nan)
        mask = np.ma.getmask(bldgs)
        labeled_mask, num_features = ndi.label(~mask)
        
        for i in range(1, num_features + 1):
            m = labeled_mask == i
            m_padded = np.pad(~m, pad_width=1, mode="constant", constant_values=True)

            shifted_up = m_padded[2:, 1:-1]
            shifted_down = m_padded[:-2, 1:-1]
            shifted_left = m_padded[1:-1, 2:]
            shifted_right = m_padded[1:-1, :-2]
            original_mask_trimmed = m_padded[1:-1, 1:-1]

            offset_mask = ~(original_mask_trimmed & shifted_up & shifted_down & shifted_left & shifted_right)
            max_bldg[m] = np.nanmax(data[offset_mask])

        return max_bldg
        
    def compute_Hs(self, H: np.ndarray) -> float:
        """
        Computes significant wave height (mean of highest 1/3) from array of wave heights.
        """
        if len(H) == 0:
            return 0.0
        H_one_third = np.quantile(H, q=2/3)
        H_top = H[H >= H_one_third]
        if len(H_top) == 0:
            return 0.0

        return float(np.mean(H_top))

    def cartesian_to_nautical_angle(self, deg: float) -> float:
        """
        Converting from cartesian to nautical angles.
        Cartesian: 0 is East, positive CCW.
        Nautical: 0 is North, positive CW.
        """
        if 0 <= deg <= 270:
            return 270.0 - deg
        elif 270 < deg < 360:
            return (270.0 - deg) + 360.0
        else:
            raise ValueError(f"{deg} must be between 0 and 360.")

    def nautical_to_xbeach_angle(self, deg: float, alfa: float) -> float:
        """
        Converting from nautical to xbeach angle using shoreline orientation alfa.
        """
        deg = deg + alfa 
        if deg >= 360:
            deg -= 360
        elif deg < 0:
            deg += 360
        return deg

    def make_directory(self, path_out: Union[str, Path]) -> str:
        """
        Make directory if it doesn't exist.
        """
        p = Path(path_out)
        p.mkdir(parents=True, exist_ok=True)
        return str(p)

    def var2label(self, var: str) -> Tuple[str, str, Any]:
        """
        Returns label, units, and color for a given variable.
        """
        v2l = { "el":"Water Elevation",
                "hs": "Sig. Wave Height",
                "Tp": "Peak Period",
                "wavedir": "Wave Direction",
                "zs": "Water Elevation",
                "zs0": "Surge Level",
                "zs1": "Water Elevation Above Surge",
                "current": "Current Velocity",
                "uu": "current x",
                "vv": "current y",
        }
        v2y = { "el": "Water Elevation (m)",
                "hs": "Sig. Wave Height (m)",
                "Tp": "Peak Period (s)",
                "wavedir": "Wave Direction",
                "zs": "Water Elevation (m)",
                "zs0": "Surge Level (m)",
                "zs1": "Water Elevation Above Surge (m)",
                "current": "Current Velocity (m/s)",
                "uu": "current x",
                "vv": "current y",
        }
        
        keys = list(v2l.keys())
        if var not in keys:
            return var, var, "black"
            
        c_idx = keys.index(var)
        colors = sns.color_palette("crest", n_colors=len(keys))
        return v2l[var], v2y[var], colors[c_idx]

    def xbeach_duration_to_start_stop(self, duration: float) -> Tuple[float, float]:
        """
        Maps simulation duration to recommended start/stop analysis times.
        """
        duration_to_start_stop = {
                    0.5: {"start": 66.25, "stop":  66.75},
                    1:   {"start": 66,    "stop":  67},
                    2:   {"start": 65.25, "stop":  67.25},
                    3:   {"start": 65,    "stop":  68},
                    4:   {"start": 64,    "stop":  68},
                    6:   {"start": 63,    "stop":  69},
                    7:   {"start": 63,    "stop":  70},
                    8:   {"start": 62,    "stop":  70},
                    10:  {"start": 61,    "stop":  71},
                    12:  {"start": 60,    "stop":  72},
                    16:  {"start": 58,    "stop":  74}
        }
        if duration not in duration_to_start_stop:
            return 0.0, 0.0 # Or some default
            
        t_start = duration_to_start_stop[duration]["start"]
        t_stop = duration_to_start_stop[duration]["stop"]
        return t_start, t_stop

    def remove_frame(self, ax: plt.Axes) -> None:
        """
        Removes spines and ticks from an axis.
        """
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.get_xaxis().set_ticks([])
        ax.get_yaxis().set_ticks([])

    def check_domain_size_wave_stat(self, run1_max: np.ndarray, run2_max: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Checks that the domain size of two runs is identical. Resizes if necessary.
        """
        if run1_max.shape != run2_max.shape:
            # Simple resizing logic from original code
            r1_to_r2 = np.array(run1_max.shape) // np.array(run2_max.shape)
            r2_to_r1 = np.array(run2_max.shape) // np.array(run1_max.shape)
            
            if np.all(r1_to_r2 > 0):
                run2_max = np.repeat(np.repeat(run2_max, r1_to_r2[0], axis=0), r1_to_r2[1], axis=1)
            elif np.all(r2_to_r1 > 0):
                run1_max = np.repeat(np.repeat(run1_max, r2_to_r1[0], axis=0), r2_to_r1[1], axis=1)

        return run1_max, run2_max

    def set_hotstart_runs(self) -> List[str]:
        """
        Finds all hotstart directories in the model path.
        """
        model_path = Path(self.path_to_model)
        if not model_path.exists():
            return []
        hotstart_runs = [d.name for d in model_path.iterdir() 
                         if d.is_dir() and d.name.startswith("hotstart_")]
        return sorted(hotstart_runs)

    def rmse(self, predictions: np.ndarray, targets: np.ndarray) -> float:
        """
        Calculates Root Mean Square Error.
        """
        return float(np.sqrt(((predictions - targets) ** 2).mean()))

    def mae(self, predictions: np.ndarray, targets: np.ndarray) -> float:
        """
        Calculates Mean Absolute Error.
        """
        return float(np.mean(np.abs(predictions - targets)))

    def get_elevated_bldgs(self, bldgs_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Separates elevated from non-elevated buildings.
        """
        if bldgs_df.index.name != "VDA_id" and "VDA_id" in bldgs_df.columns:
            bldgs_df = bldgs_df.set_index("VDA_id")

        elevated_mask = (bldgs_df["FFE_elev_status"] == "elevated") & (bldgs_df["FFE_foundation"] == "Piles/Columns")
        elevated_bldgs = bldgs_df[elevated_mask].copy()
        not_elevated = bldgs_df[~elevated_mask].copy()
        
        elevated_bldgs["elevated"] = True
        not_elevated["elevated"] = False
        
        return elevated_bldgs, not_elevated

    def compute_velocity_mag(self, ue: np.ndarray, ve: np.ndarray, return_max: bool = True) -> Union[float, np.ndarray]:        
        """
        Computes velocity magnitude from x and y components.
        """
        mag = np.sqrt(np.square(ue) + np.square(ve))
        if np.all(np.isnan(mag)):
            return np.nan
        elif return_max:
            return float(np.nanmax(mag))
        else:
            return mag

    def compute_velocity_dir(self, ue: np.ndarray, ve: np.ndarray) -> float:
        """
        Computes velocity direction at the time of maximum magnitude.
        """
        mag = np.sqrt(np.square(ue) + np.square(ve))
        if np.all(np.isnan(mag)):
            return np.nan
        max_idx = np.nanargmax(mag)
        return float(self.compute_angle(ue[max_idx], ve[max_idx]))

    def compute_angle(self, x_vec: Union[float, np.ndarray], y_vec: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Calculates the angle [0, 360) of a 2D vector.
        """
        angles_rad = np.arctan2(y_vec, x_vec)
        angles_deg = np.degrees(angles_rad)
        return np.where(angles_deg < 0, angles_deg + 360, angles_deg).item() if np.isscalar(angles_deg) else np.where(angles_deg < 0, angles_deg + 360, angles_deg)

    def calculate_running_avg(self, time_sec: np.ndarray, values: np.ndarray, window_sec: float, new_step_sec: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculates a running average and optionally resamples.
        """
        time_index = pd.to_timedelta(time_sec, unit='s')
        ts = pd.Series(values, index=time_index)
        rolling_avg = ts.rolling(window=f'{window_sec}s', min_periods=1).mean()

        if new_step_sec is not None:
            rolling_avg = rolling_avg.resample(f'{new_step_sec}s').nearest()

        return rolling_avg.index.total_seconds().to_numpy(), rolling_avg.to_numpy()

if __name__ == '__main__':
    hf = HelperFuncs()


















