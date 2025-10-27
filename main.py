import os
import matplotlib.pyplot as plt
from helpers.helpers import HelperFuncs
from make_animation.make_animation import MakeAnimation


# from compare_forcing_output.compare_forcing_output import CompareForcingOutput
from plot_forcing.plot_forcing import PlotForcing
from plot_grid.plot_grid import PlotGrid
# from plot_high_water_marks.plot_high_water_marks import PlotHighWaterMarks
from plot_output_point.plot_output_point import PlotOutputPoint
from plot_output_transect.plot_output_transect import PlotOutputTransect
# from save_wave_stats.save_wave_stats import SaveWaveStats
from plot_wave_height_domain.plot_wave_height_domain import PlotWaveHeightDomain
from plot_wave_height_bldg.plot_wave_height_bldg import PlotWaveHeightBldg
from plot_wave_height_scatter.plot_wave_height_scatter import PlotWaveHeightScatter
from plot_wave_height_hist.plot_wave_height_hist import PlotWaveHeightHist
from plot_wave_heights.plot_wave_heights import PlotWaveHeights


if __name__ == "__main__":
    # # -- save wave stats
    # sws = SaveWaveStats()
    # sws.save("zs1", "Hs")
    # sws.geolocate(stat="Hmax")
    # path_to_bldgs = os.path.join(os.getcwd(), "..", "data", "buildings", "amini-bldgs-microgrid.geojson")
    # sws.assign_to_bldgs(stats=["Hs", "Hmax"], 
    #                     path_to_bldgs=path_to_bldgs, 
    #                     runs=["run52", "run53"], 
    #                     col_names=["Hs_no_bldgs", "Hs_bldgs_on_grnd", "Hs_remove_elevated", "Hmax_no_bldgs", "Hmax_bldgs_on_grnd", "Hmax_remove_elevated"]
    #                     )

    # -- animation plots
    ma = MakeAnimation(
                var              = "zs",                       # variable to plot (H=wave height; zs=water level)
                tstart           = 0,                           # start time for animation in hours; None starts at begining of simulation; in XBeach time 
                tstop            = 1,                         # end time for animation in hours; None ends at last time step in xboutput.nc; in XBeach time
                domain_size      = "estero",                     # either "estero" or "micro" for full estero island runs or very small grid respectively
                xbeach_duration  = 0.5,                           # xbeach simulation duration; used to map water elevation forcing plot to XBeach time step.
                vmin             = 2,                           # vmin for plotting
                vmax             = 4,                           # vmax for plotting
                make_all_figs    = True,                        # create all frames, or read from existing `temp` dir
                dpi              = 200,                         # image resolution (dpi = dots per inch)
                fps              = 10,
                detrend          = False,                        # detrend the elevation data
                )
    # ma.make_animation(parallel=True, num_proc=2)
    ma.plot_frame(t_hr=1)


    # # -- compare forcing to output
    # cfo = CompareForcingOutput(var="zs1", xb_locs=[5], domain="micro")
    # cfo.compare_forcing2output()

    # # -- plot forcing
    # pf = PlotForcing()
    # pf.plot(var="hs", savepoint=1, duration=2)
    # pf.plot(var="hs", savepoint=3, duration=2)
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
    # pot.plot_transect_Hs(y_trans=[10, 50, 400],
    #                     fulldomain=False,
    #                     drawdomain=False,
    #                     fname=None
    #                     )
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

    

    # # # -- plot wave height domain
    # pwhd = PlotWaveHeightDomain()
    # pwhd.plot(stat="Hs",
    #         vmin=0,
    #         vmax=1,
    #         single_frame=True,
    #         domain_size="micro",
    #         plt_bldgs=True,
    #         fname="Hs-domain.png"
    #         )
    # pwhd.plot_diff(stat="Hs",
    #         comparison_run="frun57-20p-v2",
    #         domain_size="estero",
    #         vmax=0.1,
    #         fname="Hs-diff-20p20p"
    #         )

    # -- plot wave height building
    # pwhb = PlotWaveHeightBldg()
    # pwhb.plot(stat="Hs",
    #         model_runname_w_bldgs=None, 
    #         vmax=1, 
    #         vmin=0, 
    #         domain_size="micro", 
    #         grey_background=True, 
    #         fname="Hs-bldg.png"
    #         )


    # # -- PlotWaveHeightScatter
    # pwhs = PlotWaveHeightScatter()
    # pwhs.scatter_bldg(stat="Hs", runs=["run49", "run46"], plot_hist=True, run_w_bldgs="run45", labels=["0.5 m", "1 m", "2 m"], fname="resolution-scatter-bldg.png")
    # pwhs.scatter_domain(stat="Hs", runs=["run49", "run46"], plot_hist=True, labels=["0.5 m", "1 m", "2 m"], fname="resolution-scatter-domain.png")

    # # -- PlotWaveHeightHist
    # pwhw = PlotWaveHeightHist()
    # pwhw.plot(stat="Hs", runs=["run42"], labels=["run40", "run41"]) #, fname="hist")

    plt.show()
















