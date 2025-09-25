import matplotlib.pyplot as plt
from helpers.helpers import HelperFuncs
from make_animation.make_animation import MakeAnimation


# from compare_forcing_output.compare_forcing_output import CompareForcingOutput
from plot_forcing.plot_forcing import PlotForcing
from plot_grid.plot_grid import PlotGrid
from plot_high_water_marks.plot_high_water_marks import PlotHighWaterMarks
from plot_output_point.plot_output_point import PlotOutputPoint
from plot_output_transect.plot_output_transect import PlotOutputTransect
from plot_wave_heights.plot_wave_heights import PlotWaveHeights


if __name__ == "__main__":
    # # -- animation plots
    # ma = MakeAnimation(
    #             var              = "zs1",                         # variable to plot (H=wave height; zs=water level)
    #             tstart           = 1,                           # start time for animation in hours; None starts at begining of simulation; in XBeach time 
    #             tstop            = 1.05,                         # end time for animation in hours; None ends at last time step in xboutput.nc; in XBeach time
    #             domain_size      = "micro",                     # either "estero" or "micro" for full estero island runs or very small grid respectively
    #             xbeach_duration  = 2,                           # xbeach simulation duration; used to map water elevation forcing plot to XBeach time step.
    #             vmin             = -1,                           # vmin for plotting
    #             vmax             = 1,                           # vmax for plotting
    #             make_all_figs    = True,                        # create all frames, or read from existing `temp` dir
    #             dpi              = 200,                         # image resolution (dpi = dots per inch)
    #             )
    # # ma.make_animation(parallel=True, num_proc=2)
    # ma.plot_frame(t_hr=1)


    # # -- compare forcing to output
    # cfo = CompareForcingOutput(var="zs1", xb_locs=[5], domain="micro")
    # cfo.compare_forcing2output()

    # # -- plot forcing
    # pf = PlotForcing()
    # pf.plot(var="hs", savepoint=5, duration=2)

    # # -- plot grid
    # pg = PlotGrid()
    # pg.plot_dep_across_y(x_trans=[10, 100, 200], lookonshore=True, drawdomain=True)

    # # -- plot high water marks
    # phwm = PlotHighWaterMarks()
    # phwm.plot_scatter()

    # # -- plot output point
    # pop = PlotOutputPoint()
    # pop.plot(var="zs1",
    #         xys=[[4,0], [200,400], [360,400], [373,400], [600,400]], 
    #         drawdomain=True,
    #         fulldomain=False, 
    #         savefig=True
    #         )

    # # -- plot transect
    # pot = PlotOutputTransect()
    # pot.plot_water_level_transect(var="zs1", y_trans=400,
    #                              ts=[1],
    #                              h_plus_zs=False,
    #                              drawdomain=True, 
    #                              fulldomain=False,
    #                              savefig=True
                                 # )
    # pot.video_transect(var="zs1", 
    #                   y_trans=400,
    #                   t_start=1,
    #                   t_stop=1.01,
    #                   h_plus_zs=False,
    #                   dpi=100,
    #                   )

    
    # # -- plot wave heights
    pwh = PlotWaveHeights()
    pwh.




    plt.show()
















