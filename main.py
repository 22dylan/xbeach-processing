import os
import matplotlib.pyplot as plt
from helpers.helpers import HelperFuncs


from make_animation.make_animation import MakeAnimation
from make_animation_hotstart.make_animation_hotstart import MakeAnimationHotstart
# from compare_forcing_output.compare_forcing_output import CompareForcingOutput
from plot_bldg_dmg.plot_bldg_dmg import PlotBldgDmg
from plot_forcing.plot_forcing import PlotForcing
from plot_grid.plot_grid import PlotGrid
from plot_high_water_marks.plot_high_water_marks import PlotHighWaterMarks
from plot_output_point.plot_output_point import PlotOutputPoint
from plot_output_transect.plot_output_transect import PlotOutputTransect
from save_wave_stats.save_wave_stats import SaveWaveStats
from plot_violin_dmg.plot_violin_dmg import PlotViolinDmg
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
    #          # stats=["Hmax", "Hs_max", "Hs_tot", "zs_max", "t_Hs_1m", "t_Hs_2m", "t_Hs_3m"],
    #          stats = ["impulse"],
    #          trim_beginning_seconds=500, 
    #          store_in_mem=False,
    #          chunk_size_min=15,
    #          max_workers=100,
    #          )

    # sws.geolocate(stat="impulse")

    # sws.assign_to_bldgs(stats=["Hmax", "Hs_max", "Hs_tot", "impulse", "surge_max", "t_Hs_0.5m", "t_Hs_0.25m", "t_Hs_1m", "t_Hs_2m", "t_Hs_3m", "zs_max", "zs_mean"],
    #                     # runs=["run62"],
    #                     col_names=["Hmax", "Hs_max", "Hs_tot", "impulse", "surge_max", "t_Hs_0.5m", "t_Hs_0.25m", "t_Hs_1m", "t_Hs_2m", "t_Hs_3m", "zs_max", "zs_mean"]
    #                     )


    # -- animation plots
    # ma = MakeAnimation(
    #             var              = "zs",                        # variable to plot (H=wave height; zs=water level)
    #             tstart           = 179/60,                           # start time for animation in hours; None starts at begining of simulation; in XBeach time 
    #             tstop            = 181/60,                        # end time for animation in hours; None ends at last time step in xboutput.nc; in XBeach time
    #             domain_size      = "micro",                     # either "estero" or "micro" for full estero island runs or very small grid respectively
    #             xbeach_duration  = 0.5,                         # xbeach simulation duration; used to map water elevation forcing plot to XBeach time step.
    #             vmin             = 0,                           # vmin for plotting
    #             vmax             = 5,                           # vmax for plotting
    #             make_all_figs    = True,                        # create all frames, or read from existing `temp` dir
    #             dpi              = 100,                         # image resolution (dpi = dots per inch)
    #             fps              = 10,
    #             detrend          = False,                       # detrend the elevation data
    #             )

    # ma.make_animation(parallel=True, num_proc=1)
    # ma.plot_frame(t_hr=3600/3600)



    # -- animation for hotsttart runs
    ma = MakeAnimationHotstart(
                var              = "zs",                        # variable to plot (H=wave height; zs=water level)
                tstart           = 239/60,                           # start time for animation in hours; None starts at begining of simulation; in XBeach time 
                tstop            = 241/60,                        # end time for animation in hours; None ends at last time step in xboutput.nc; in XBeach time
                domain_size      = "micro",                     # either "estero" or "micro" for full estero island runs or very small grid respectively
                xbeach_duration  = 0.5,                         # xbeach simulation duration; used to map water elevation forcing plot to XBeach time step.
                vmin             = 0,                           # vmin for plotting
                vmax             = 5,                           # vmax for plotting
                make_all_figs    = True,                        # create all frames, or read from existing `temp` dir
                dpi              = 100,                         # image resolution (dpi = dots per inch)
                fps              = 10,
                detrend          = False,                       # detrend the elevation data
                dt_video         = 2,
                )

    hotstart_runs = ["test_a1", "test_a2", "test_b1", "test_b2", "test_c1", "test_c2", 
                     "test_d1", "test_d2", "test_e1", "test_e2", "test_f1", "test_f2", 
                     "test_g1", "test_g2", "test_h1", "test_h2"]
    ma.make_animation_hotstart(hotstart_runs=hotstart_runs)


    # # -- compare forcing to output
    # cfo = CompareForcingOutput(var="zs1", xb_locs=[5], domain="micro")
    # cfo.compare_forcing2output()

    # # -- plot forcing
    # pf = PlotForcing()
    # pf.plot(var="hs", savepoint=1, figsize=(10,3), duration=2, fname="hs-2hr")

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
    #         xys=[[200,400]], 
    #         drawdomain=True,
    #         fulldomain=False, 
    #         savefig=True
    #         )
    # pop.plot_Hs(var="zs",
    #         xys=[[200,400]], 
    #         chunk_size_min=15,
    #         drawdomain=False,
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
    # pwhd = PlotWaveHeightDomain()
    # pwhd.plot(stat="surge_max",
    #         vmin=3,
    #         vmax=5,
    #         single_frame=True,
    #         domain_size="estero",
    #         plt_bldgs=True,
    #         plt_offshore=True,
    #         fname="surge_max-domain.png"
    #         )
    # pwhd.plot_diff(stat="Hs",
    #         comparison_run="run64-nowind",
    #         domain_size="estero",
    #         vmax=0.5,
    #         fname="Hs-diff-windnowind"
    #         )

    # # # -- plot wave height building
    # pwhb = PlotWaveHeightBldg()
    # pwhb.plot(stat="surge_max",
    #         model_runname_w_bldgs=None,
    #         vmin=0,
    #         vmax=None,
    #         domain_size="estero", 
    #         grey_background=False, 
    #         # fname="Hmax-bldg.png"
    #         )

    # -- PlotWaveHeightScatter
    # pwhs = PlotWaveHeightScatter()
    # pwhs.scatter_bldg(stat="Hs", runs=["run60_45p_2Hz", "run60_45p_1Hz"], plot_hist=True, run_w_bldgs="run60_45p_4Hz", labels=["4 Hz", "2 Hz", "1 Hz"], fname="samplingfreq-scatter-bldg.png")
    # pwhs.scatter_domain(stat="Hs", runs=["run60_45p_2Hz", "run60_45p_1Hz"], plot_hist=True, labels=["4 Hz", "2 Hz", "1 Hz"], fname="samplingfreq-scatter-domain.png")

    # pwhe = PlotWaveHeightError()
    # pwhe.plot_error(stat="Hs", runs=["s16", "s8", "s4", "s2"], run_w_bldgs="s1", labels=["s1", "s16", "s8", "s4", "s2"], fname="temp-error.png")

    # # -- PlotWaveHeightHist
    # pwhw = PlotWaveHeightHist()
    # pwhw.plot(stat="Hs", runs=["run42"], labels=["run40", "run41"]) #, fname="hist")


    # # -- PlotBldgDmg
    # pbd = PlotBldgDmg()
    # pbd.plot()


    # # -- PlotViolinDmg
    # pvd = PlotViolinDmg()
    # path_to_stats = os.path.join(os.getcwd(), "..", "processed-results", "run62", "H_at_bldgs.csv")
    # pvd.plot(stats=["impulse" , "Hs_max", "Hs_tot", "Hmax", "surge_max", "t_Hs_0.5m"], 
    #         path_to_stats=path_to_stats,
    #         ncols=2,
    #         fname="violin.png"
    #         )







    plt.show()

















