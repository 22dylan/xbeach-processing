import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from helpers.helpers import HelperFuncs


class PlotWaveHeightDomain(HelperFuncs):
    """docstring for plot_wave_heights"""
    def __init__(self):
        super().__init__()

    def plot(self, stat, vmax=None, vmin=None, fname=None, prnt_read=False, 
            single_frame=False, domain_size="estero", plt_bldgs=True):
        
        # read wave heights
        H = self.read_npy(stat)

        # read buildings and grid 
        xgr, ygr, zgr = self.read_grid()
        bldgs = self.read_buildings()
        mask = np.ma.getmask(bldgs)
        figsize = self.get_figsize(domain_size)

        # fig, ax = plt.subplots(1,1, figsize=figsize)
        if single_frame:
            fig, ax0 = plt.subplots(1,1, figsize=figsize)
        else:
            fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(16,9), gridspec_kw={'width_ratios': [1,2.8]})

        # setting up mask to ignore values less than 0
        mask = (zgr<=0)
        masked_array = np.ma.array(H, mask=mask)

        # setting up colormap for water
        if stat == "H":
            cmap = mpl.cm.plasma
            cmap.set_bad('grey')
            vmax = 1.0 if vmax == None else vmax
            vmin = 0
        else:
            cmap = mpl.cm.plasma
            # cmap = mpl.cm.cividis
            cmap.set_bad('grey')
            vmax = 1 if vmax == None else vmax
            vmin = 0 if vmin == None else vmin

        # -- drawing first plot
        pcm = ax0.pcolormesh(xgr, ygr, masked_array, vmin=vmin, vmax=vmax, cmap=cmap)
        if single_frame:
            ax_bar = ax0
        else:
            ax_bar = ax1
        plt.colorbar(pcm, ax=ax_bar, extend="max", label="Sig. Wave Height (m)", aspect=40)
        if plt_bldgs:
            custom_color = 'springgreen'
            cmap_bldg = mpl.colors.ListedColormap([custom_color])
            cmap_bldg.set_bad(alpha=0)

            ax0.pcolormesh(xgr, ygr, bldgs, cmap=cmap_bldg)
        ax0.set_xlabel("x (m)")
        ax0.set_ylabel("y (m)")

        if single_frame==False:
            # -- drawing second, zoomed in plot
            # full model domain
            box_lower_left = (2600, 5000)       # in world units
            dx, dy = 1000, 1000
            # continuing with zommed in plot
            box_upper_right = (box_lower_left[0]+dx, box_lower_left[1]+dy)

            id_ll = self.xy_to_grid_index(xgr, ygr, box_lower_left)
            id_ur = self.xy_to_grid_index(xgr, ygr, box_upper_right)
            
            xgr2 = xgr[id_ll[1]:id_ur[1], id_ll[0]:id_ur[0]]
            ygr2 = ygr[id_ll[1]:id_ur[1], id_ll[0]:id_ur[0]]
            masked_array2 = masked_array[id_ll[1]:id_ur[1], id_ll[0]:id_ur[0]]
            bldgs2 = bldgs[id_ll[1]:id_ur[1], id_ll[0]:id_ur[0]]

            ax1.pcolormesh(xgr2, ygr2, masked_array2, vmin=vmin, vmax=vmax, cmap=cmap)
            ax1.pcolormesh(xgr2, ygr2, bldgs2, cmap=cmap_bldg)

            # # --old
            box_l = xgr2[0,-1] - xgr2[0,0]
            box_h = ygr2[-1,0] - ygr2[0,0]

            # # -- adding rectangle showing where zoomed in area is
            rect = patches.Rectangle(box_lower_left, box_l, box_h, linewidth=3, zorder=10, edgecolor='r', facecolor='none')
            ax0.add_patch(rect)

        # --- saving file
        self.save_fig(fig, fname, transparent=True, dpi=300)


    def plot_diff(self, stat, comparison_run, vmax=1, norm=False, 
                domain_size="estero", fname=None):
        
        # get difference in wave heights
        run1_max = self.read_npy(stat)
        run2_max = self.read_npy(stat, comparison_run)

        mask = np.isnan(run1_max) & np.isnan(run2_max)
        run1_max = np.ma.array(run1_max, mask=mask) # here mask tells numpy which cells to ignore.
        run2_max = np.ma.array(run2_max, mask=mask) # here mask tells numpy which cells to ignore.

        if norm == True:
            denom = (run1_max+run2_max)/2
            diff = ((run1_max - run2_max)/denom)
        else:
            diff = run1_max - run2_max

        # -- read in grid
        xgr, ygr, zgr = self.read_grid()
        bldgs = self.read_buildings()

        figsize = self.get_figsize(domain_size)
        fig, ax0 = plt.subplots(1,1, figsize=figsize)

        # determine where water is and setup mask to ignore those cells.
        mask = (zgr<0)
        masked_array = np.ma.array(diff, mask=mask) # here mask tells numpy which cells to ignore.
        
        # cmap = mpl.cm.PiYG
        cmap = mpl.cm.PuOr
        cmap.set_bad('bisque')
        vmin = -vmax

        cmap_bldg = mpl.cm.Greys_r
        cmap_bldg.set_bad(alpha=0)

        # -- drawing first plot
        if norm:
            label = "Percent Difference"
            vmax = 1
            vmin = -1
            extend = None
        else:
            label = "Difference (m)"
            extend = "both"
        pcm = ax0.pcolormesh(xgr, ygr, masked_array, vmin=vmin, vmax=vmax, cmap=cmap)
        plt.colorbar(pcm, ax=ax0, extend=extend, label=label, aspect=40)
        ax0.pcolormesh(xgr, ygr, bldgs, cmap=cmap_bldg)
        ax0.set_xlabel("x (m)")
        ax0.set_ylabel("y (m)")

        # --- saving file
        self.save_fig(fig, fname, transparent=True, dpi=300)




