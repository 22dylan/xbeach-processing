# xbeach-plotting
Scripts for processing and plotting XBeach output.


To use this code locally, follow the steps below: 
1. Decide where you want to put the python files on your computer, and run the command below. A new directory will be written called `xbeach-processing/`
    1. `git clone https://github.com/22dylan/xbeach-processing.git`
2. In xbeach-processing/, there’s a file called “paths.txt”. You’ll need to set three paths here:
    1. `path_to_model`: This is where the XBeach run you want to process is located. You just need to set the directory, not the `xboutput.nc` file.
    2. `path_to_forcing`: This is the water-elevation file that is used as input to XBeach. This is needed if you are going to make a movie using `make_animation`. The water elevation time series is read from this file and plotted.
    3. `path_to_save_plot`: This is where to save the video/figures. 
3. In `xbeach-processing/`, there’s a file named `main.py`. 
    1. Uncomment the function corresponding to the figure/video you want to make. 
4. The resulting figure/video will be written to the path that you specified under `path_to_save_plot/`