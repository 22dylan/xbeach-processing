from helpers.helpers import HelperFuncs
from compare_forcing_output.compare_forcing_output import CompareForcingOutput
from make_animation.make_animation import MakeAnimation



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


    cfo = CompareForcingOutput()
    cfo.compare_forcing2output()