import pandas as pd
import matplotlib.pyplot as plt

from helpers.helpers import HelperFuncs

class PlotWaveHeightHist(HelperFuncs):
    """docstring for plot_wave_heights"""
    def __init__(self):
        super().__init__()

    def plot(self, stat, runs, labels, run_w_bldgs=None, fname=None):        
        df = pd.DataFrame()
        runs.insert(0, self.model_runname)
        for run in runs:
            # read wave heights
            H = self.read_npy(stat, run)
            df[run] = H.flatten()

        print(df.min())

        # --- plotting histogram - comparing max wave heights, same plot
        fig, ax = plt.subplots(1,1, figsize=(8,4.5))
        colors = ["tan", "darkslategrey", "olive"]
        zorders = [1, 0, 2]
        alphas = [0.7, 0.7, 0.7]
        for run_i, run in enumerate(df.columns):
            df[run].hist(ax=ax, 
                        bins=100, 
                        range=(0,2),
                        density=False, 
                        zorder=zorders[run_i], 
                        color=colors[run_i], 
                        edgecolor='black', 
                        linewidth=0.3,
                        label=labels[run_i],
                        alpha=alphas[run_i]
                        )

        ax.grid(False)
        ax.legend(bbox_to_anchor=(0.9, 0.95), frameon=False, facecolor=None)
        ax.set_xlabel("Sig. Wave Height (m)")
        ax.set_ylabel("Frequency")
        ax.set_xlim([0,2])
        # ---
        self.save_fig(fig, fname, transparent=True, dpi=300)


    plt.show()

