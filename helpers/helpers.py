import os
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

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
        self.project_dir = os.path.join(self.file_dir, "..")
        self.read_paths()
        self.xboutput_filename = self.get_output_filename()

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
                    relative_path = bool(line.split("=")[1])

                if line and '=' in line:
                    key, value = line.split('=', 1)
                    key = key.strip()
                    value = value.strip()
                    value = value.split("/")
                    if relative_path==True:
                        pth = os.path.join(self.project_dir, *value)                
                    else:
                        pth = os.path.join(*value)
                    setattr(self, key, pth)
                    print("  successfully set {}" .format(key, pth))
        print(self.path_to_model)

    def get_output_filename(self):
        """
        Returns the output file name that is located in the `path_to_model` 
          directory.
        """
        files = os.listdir(self.path_to_model)
        fn  = [i for i in files if ".nc" in i][0]
        return fn

    def read_2d_data_xarray_timestep(self, var, t):        
        """
        Reads xarray data for entire domain at specified time step.
        Inputs:
            var: variable to read, e.g., zs1
            t: time step to read
        Returns: 
            data: 2D numpy array.
        """
        fn = os.path.join(self.path_to_model, self.xboutput_filename)
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

    def read_buildings(self):
        """
        TODO: add docstring
        """
        fn_zgrid = os.path.join(self.path_to_model, "z.grd")
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

    def read_grid(self):
        """
        TODO: add docstring
        """
        # -- reading xgrid
        xgrid = os.path.join(self.path_to_model, "x.grd")
        with open(xgrid,'r') as f:
            for cnt, line in enumerate(f.readlines()):
                xs = [float(i.strip()) for i in line.split()]
                if cnt == 0:
                    break
        
        # -- reading ygrid
        ys = []
        ygrid = os.path.join(self.path_to_model, "y.grd")
        with open(ygrid,'r') as f:
            for cnt, line in enumerate(f.readlines()):
                y_ = [float(i.strip()) for i in line.split()][0]
                ys.append(y_)
        
        # -- reading zgrid
        zgr = np.zeros((len(ys), len(xs)))
        zgrid = os.path.join(self.path_to_model, "z.grd")
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

    def save_fig(self, fig, fn=None, **kwargs):
        """
        Saves figures
        Inputs:
            fig: matplotlib figure. 
            fn: filename; if None, no figure is created
            **kwargs: keyword args to pass to matplotlib savefig function. 
        """
        if fn != None:
            plt.savefig(fn,
                        **kwargs,
                        # transparent=True, 
                        # dpi=self.dpi,
                        # bbox_inches='tight',
                        # pad_inches=0.1,
                        )
            plt.close()


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
            os.makedirs(path_out)
        return path_out


if __name__ == '__main__':
    hf = HelperFuncs()


