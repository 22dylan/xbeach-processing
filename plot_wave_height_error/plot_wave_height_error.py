import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.ndimage as ndi

from helpers.helpers import HelperFuncs

class PlotWaveHeightError(HelperFuncs):
    """docstring for plot_wave_heights"""
    def __init__(self):
        super().__init__()

    def plot_error(self, stat, runs, labels, run_w_bldgs, plot_hist=True, lim=3, fname=None):
        runs.insert(0, self.model_runname)
        sizes = []
        for run in runs:
            H = self.read_npy(stat, run)
            sizes.append(np.shape(H)[0])
        largest_size_idx = np.argmax(sizes)

        wave_heights = {}
        for run in runs:
            H = self.read_npy(stat, run)                
            if run == "s1":
                print("This is crudely done")
                first_row = H[0:1, :]
                last_row =  H[-1:, :]
                H = np.vstack((first_row, H, last_row))
                H = H[:,:-2]
            elif run == "s2":
                print("This is crudely done")
                first_row = H[0:1, :]
                H = np.vstack((first_row, H))
                H = H[:,:-1]

            wave_heights[run] = H

        H_domain = wave_heights[runs[largest_size_idx]]
        for run in runs:
            _, wave_heights[run] = self.check_domain_size_wave_stat(H_domain, wave_heights[run])

        df = pd.DataFrame()
        for run in runs:
            df[run] = wave_heights[run].flatten()

        # -- now switch to height at buildings
        # -- getting buildings / location of buildings
        bldgs = self.read_buildings(run_w_bldgs)
        print("this is hideously done")
        first_row = bldgs[0:1, :]
        last_row =  bldgs[-1:, :]
        bldgs = np.vstack((first_row, bldgs, last_row))
        bldgs = bldgs[:,:-2]
        mask = (bldgs != 10)
        bldgs = np.ma.array(bldgs, mask=mask)
        mask_bldgs = np.ma.getmask(bldgs)


        df_bldgs = pd.DataFrame()
        for run in runs:
            H = wave_heights[run]
            bldg_H = self.assign_max_to_bldgs(H, bldgs)    # getting max at each building

            labeled_mask, num_features = ndi.label(~mask_bldgs)
            max_H_at_bldgs = []
            for i in range(num_features+1):
                if i == 0:
                    continue
                m_ = labeled_mask==i
                val = np.max(bldg_H[m_])
                max_H_at_bldgs.append(val.item())
            df_bldgs[run] = max_H_at_bldgs

        """ removing any row that is 0, this happens when comparing runs at
            different resolutions - wave heights at the buildings get funny
        """
        df = df[~(df == 0).any(axis=1)]

        x_data = df[runs[0]]
        x_data_bldg = df_bldgs[runs[0]]
        RMSEs, RMSEs_bldg = [], []
        MAEs, MAEs_bldg = [], []
        for run in runs[1:]:
            y_data = df[run]
            y_data_bldg = df_bldgs[run]

            rmse = self.rmse(x_data, y_data)
            mae  = self.mae(x_data, y_data)
            RMSEs.append(rmse)
            MAEs.append(mae)

            rmse = self.rmse(x_data_bldg, y_data_bldg)
            mae  = self.mae(x_data_bldg, y_data_bldg)
            RMSEs_bldg.append(rmse)
            MAEs_bldg.append(mae)

        fig, ax = plt.subplots(1,1, figsize=(8,3))

        x = range(len(runs)-1)
        
        ax.scatter(x,RMSEs, color="cadetblue", )
        ax.plot(x, RMSEs,  ls="-.", color="cadetblue", label="RMSE (domain)")
        ax.scatter(x,MAEs, color="cadetblue")
        ax.plot(x, MAEs, color="cadetblue", ls="-", label="MAE (domain)")

        ax.scatter(x,RMSEs_bldg, color="chocolate")
        ax.plot(x, RMSEs_bldg,  ls="-.", color="chocolate", label="RMSE (building)")
        ax.scatter(x,MAEs_bldg, color="chocolate")
        ax.plot(x, MAEs_bldg, color="chocolate", ls="-", label="MAE (building)")

        ax.set_ylabel("Error from {} (m)" .format(labels[0]))
        
        ax.set_xticks(x)
        ax.set_ylim([0,1])
        ax.set_xticklabels(labels[1:])
        ax.legend()
        ax.grid()
        ax.set_axisbelow(True)

        self.save_fig(fig, fname, transparent=True, dpi=300)
        













