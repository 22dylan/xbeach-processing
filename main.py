import os
import matplotlib.pyplot as plt
from helpers.helpers import HelperFuncs
from make_animation.make_animation import MakeAnimation


# from compare_forcing_output.compare_forcing_output import CompareForcingOutput
from plot_forcing.plot_forcing import PlotForcing
from plot_grid.plot_grid import PlotGrid
from plot_high_water_marks.plot_high_water_marks import PlotHighWaterMarks
from plot_output_point.plot_output_point import PlotOutputPoint
from plot_output_transect.plot_output_transect import PlotOutputTransect
from save_wave_stats.save_wave_stats import SaveWaveStats
from plot_wave_height_domain.plot_wave_height_domain import PlotWaveHeightDomain
from plot_wave_height_bldg.plot_wave_height_bldg import PlotWaveHeightBldg
from plot_wave_height_error.plot_wave_height_error import PlotWaveHeightError
from plot_wave_height_scatter.plot_wave_height_scatter import PlotWaveHeightScatter
from plot_wave_height_hist.plot_wave_height_hist import PlotWaveHeightHist
from plot_wave_heights.plot_wave_heights import PlotWaveHeights

if __name__ == "__main__":
    # -- save wave stats
    # sws = SaveWaveStats()
    # sws.save(var="zs",
    #          stats=["Hmax", "Hs_max", "Hs_tot", "zs_max", "t_Hs_1m", "t_Hs_2m", "t_Hs_3m"],
    #          trim_beginning_seconds=500, 
    #          store_in_mem=False,
    #          chunk_size_min=15,
    #          max_workers=10,
    #          )
    # sws.geolocate(stat="Hmax")

    # path_to_bldgs = os.path.join(os.getcwd(), "..", "data", "buildings", "amini-bldgs-microgrid.geojson")
    # sws.assign_to_bldgs(stats=["Hs", "Hmax"], 
    #                     path_to_bldgs=path_to_bldgs, 
    #                     runs=["run52", "run53"], 
    #                     col_names=["Hs_no_bldgs", "Hs_bldgs_on_grnd", "Hs_remove_elevated", "Hmax_no_bldgs", "Hmax_bldgs_on_grnd", "Hmax_remove_elevated"]
    #                     )

    # -- animation plots
    # ma = MakeAnimation(
    #             var              = "zs1",                       # variable to plot (H=wave height; zs=water level)
    #             tstart           = 0,                           # start time for animation in hours; None starts at begining of simulation; in XBeach time 
    #             tstop            = 1,                         # end time for animation in hours; None ends at last time step in xboutput.nc; in XBeach time
    #             domain_size      = "estero",                     # either "estero" or "micro" for full estero island runs or very small grid respectively
    #             xbeach_duration  = 0.5,                           # xbeach simulation duration; used to map water elevation forcing plot to XBeach time step.
    #             vmin             = -1,                           # vmin for plotting
    #             vmax             = 1,                           # vmax for plotting
    #             make_all_figs    = True,                        # create all frames, or read from existing `temp` dir
    #             dpi              = 200,                         # image resolution (dpi = dots per inch)
    #             fps              = 10,
    #             detrend          = False,                        # detrend the elevation data
    #             )
    # ma.make_animation(parallel=True, num_proc=10)
    # ma.plot_frame(t_hr=1)

    # # -- compare forcing to output
    # cfo = CompareForcingOutput(var="zs1", xb_locs=[5], domain="micro")
    # cfo.compare_forcing2output()

    # # -- plot forcing
    # pf = PlotForcing()
    # pf.plot(var="el", savepoint=5, duration=4) #, fname="el")

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
    # pop.plot_timeseries(var="zs",
    #         xys=[[100,400]], 
    #         drawdomain=True,
    #         fulldomain=False, 
    #         savefig=False
    #         )
    # pop.plot_Hs(var="zs",
    #         xys=[[500,400]], 
    #         time_chunks_min=15,
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

    

    # # -- plot wave height domain
    pwhd = PlotWaveHeightDomain()
    pwhd.plot(stat="Hs_max",
            vmin=0,
            vmax=2,
            single_frame=True,
            domain_size="micro",
            plt_bldgs=True,
            plt_offshore=False,
            # fname="Hs-domain.png"
            )
    # pwhd.plot_diff(stat="Hs",
    #         comparison_run="run64-nowind",
    #         domain_size="estero",
    #         vmax=0.5,
    #         fname="Hs-diff-windnowind"
    #         )

    # # -- plot wave height building
    # pwhb = PlotWaveHeightBldg()
    # pwhb.plot(stat="Hs",
    #         model_runname_w_bldgs=None, 
    #         vmax=1, 
    #         vmin=0, 
    #         domain_size="micro", 
    #         grey_background=False, 
    #         fname="Hs-bldg.png"
    #         )


    # -- PlotWaveHeightScatter
    # pwhs = PlotWaveHeightScatter()
    # pwhs.scatter_bldg(stat="Hs", runs=["s2", "s4", "s8", "s16"], plot_hist=True, run_w_bldgs="s1", labels=["1 m", "2 m", "4 m", "8 m", "16 m"], fname="temp-scatter-bldg.png")
    # pwhs.scatter_domain(stat="Hs", runs=["s2", "s4", "s8", "s16"], plot_hist=True, labels=["1 m", "2 m", "4 m", "8 m", "16 m"], fname="temp-scatter-domain.png")

    # pwhe = PlotWaveHeightError()
    # pwhe.plot_error(stat="Hs", runs=["s16", "s8", "s4", "s2"], run_w_bldgs="s1", labels=["s1", "s16", "s8", "s4", "s2"], fname="temp-error.png")

    # # -- PlotWaveHeightHist
    # pwhw = PlotWaveHeightHist()
    # pwhw.plot(stat="Hs", runs=["run42"], labels=["run40", "run41"]) #, fname="hist")

    plt.show()
















