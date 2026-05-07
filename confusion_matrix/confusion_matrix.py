import os

# Confusion matrix plotting utilities
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

from helpers.helpers import HelperFuncs
from save_wave_stats.save_wave_stats import SaveWaveStats


# Class that encapsulates confusion‑matrix generation and saving
class ConfusionMatrixPlotter(HelperFuncs):
    """Utility class to generate and save a confusion matrix.
    +
    +    This class follows the same pattern as the other processing classes in the
    +    repository by inheriting from ``HelperFuncs``. The base class provides the
    +    ``path_to_save_plot`` and ``path_to_dmg`` attributes, so no explicit
    +    arguments are required when constructing the object.
    +"""

    def __init__(self):
        super().__init__()

    def plot_confusion(
        self,
        damaged_DSs=["DS6"],
        bldgs="all",
        elevated_kwds=None,
        fname=None,
    ):
        """Plot and save a confusion matrix comparing XBeach predictions to VDA observations.

        Parameters
        ----------
        damaged_DSs: list, optional
            Damage states considered as "destroyed" for the VDA observations.
        bldgs: str, optional
            Which buildings to include: "all", "non-elevated", or "elevated".
        elevated_kwds: dict, optional
            Dictionary that may contain the key ``compute_removed_elevated``. If True, the
            removed‑elevated CSV will be recomputed.
        fname: str, optional
            Filename for the saved figure (without directory). If ``None`` a default name
            ``confusion_matrix.png`` is used.
        """

        # Ensure the CSV with removed buildings exists
        fn = os.path.join(self.path_to_save_plot, "removed_bldgs_all.csv")  # type: ignore[attr-defined]
        recompute = False
        if elevated_kwds is not None:
            recompute = elevated_kwds.get("compute_removed_elevated", False)
        if (not os.path.exists(fn)) or recompute:
            sws = SaveWaveStats()
            sws.save_removed_bldgs()

        # Load XBeach predictions
        df_xbeach = pd.read_csv(fn)
        df_xbeach.set_index("VDA_id", inplace=True)

        # Filter based on building selection
        if bldgs == "all":
            txt = "All buildings (including elevated)"
        elif bldgs == "non-elevated":
            df_xbeach = df_xbeach.loc[~df_xbeach["elevated"]]
            txt = "Ignore Elevated"
        elif bldgs == "elevated":
            df_xbeach = df_xbeach.loc[df_xbeach["elevated"]]
            txt = "Elevated Only"
        else:
            raise ValueError(
                "bldgs keyword must be: `all`, `elevated` or `non-elevated`"
            )

        # Load VDA observations
        df_dmg = pd.read_csv(self.path_to_dmg)  # type: ignore[attr-defined]
        df_dmg.set_index("VDA_id", inplace=True)
        df_dmg["removed_vda"] = 0
        df_dmg.loc[df_dmg["VDA_DS_overall"].isin(damaged_DSs), "removed_vda"] = 1

        # Merge predictions and observations
        df = pd.merge(
            df_xbeach["remove"],
            df_dmg["removed_vda"],
            left_index=True,
            right_index=True,
        )

        # Compute confusion matrix
        labels = [0, 1]
        cm = confusion_matrix(df["removed_vda"], df["remove"], labels=labels)
        score = (cm[0, 0] + cm[1, 1]) / np.sum(cm)
        score_text = f"Percent Correct: {score:0.3f}"

        # Plot
        fig = plt.figure(figsize=(6, 4))
        ax = fig.add_subplot(111)
        ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Greys)  # type: ignore[attr-defined]
        ax.set_ylabel("Observed", fontsize=14, rotation=0)  # type: ignore[attr-defined]
        ax.set_xlabel("XBeach", fontsize=14)  # type: ignore[attr-defined]
        ax.text(1.0, 1.01, score_text, transform=ax.transAxes, ha="right", va="bottom")  # type: ignore[attr-defined]
        ax.text(1.0, 1.08, txt, transform=ax.transAxes, ha="right", va="bottom")  # type: ignore[attr-defined]

        class_labels = ["Standing", "Destroyed"]
        tick_marks = range(len(class_labels))
        ax.set_xticks(tick_marks)
        ax.set_xticklabels(class_labels, rotation=0, fontsize=10)
        ax.set_yticks(tick_marks)
        ax.set_yticklabels(class_labels, rotation=0, fontsize=10)

        thresh = cm.max() / 2.0
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(
                    j,
                    i,
                    cm[i, j],
                    ha="center",
                    va="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontsize=12,
                )
        ax.set_xticks(np.arange(-0.5, len(class_labels), 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(class_labels), 1), minor=True)
        ax.grid(which="minor", color="k", linestyle="-", linewidth=0.5)
        fig.tight_layout()  # type: ignore[attr-defined]

        # Save figure
        if fname is None:
            fname = "confusion_matrix.png"
        save_path = os.path.join(self.path_to_save_plot, fname)  # type: ignore[attr-defined]
        self.save_fig(save_path, dpi=1000)  # type: ignore[attr-defined]

    # Additional helper method for exploratory analysis of confusion data
    def explore_confusion(self, damaged_DSs=["DS5", "DS6"]):
        fn = os.path.join(self.path_to_save_plot, "removed_bldgs_all.csv")
        df_xbeach = pd.read_csv(fn)  # read csv
        df_xbeach = df_xbeach.loc[
            df_xbeach["remove"] != -9999
        ]  # remove buildings outside domain
        df_xbeach.set_index("VDA_id", inplace=True)  # set index
        df_xbeach = df_xbeach.loc[
            df_xbeach["elevated"] == False
        ]  # get non-elevated buildings

        df_dmg = pd.read_csv(self.path_to_dmg)  # read observations from VDA
        df_dmg.set_index("VDA_id", inplace=True)  # set index
        df_dmg["removed_vda"] = 0
        df_dmg.loc[df_dmg["VDA_DS_overall"].isin(damaged_DSs), "removed_vda"] = 1
        df_dmg = pd.merge(
            df_dmg, df_xbeach, left_index=True, right_index=True, how="right"
        )
        df_dmg = df_dmg[~df_dmg.index.duplicated(keep="first")]

        df_dmg["TA_ActYearBuilt_pre1974"] = False
        df_dmg.loc[df_dmg["TA_ActYearBuilt"] <= 1974, "TA_ActYearBuilt_pre1974"] = True
        df_dmg = df_dmg.loc[df_dmg["TA_ActYearBuilt_pre1974"] == True]
        df_dmg = df_dmg.loc[df_dmg["TA_ShapeSTArea_Sqft"] < 4000]
        df_dmg = df_dmg.loc[df_dmg["TA_BldgUseTyp"] != "mobile home"]
        # df_dmg = df_dmg.loc[df_dmg["TA_BldgUseTyp"] == "mobile home"]

        fn = os.path.join(self.path_to_save_plot, "stats_at_bldgs.csv")
        stats_at_bldgs = pd.read_csv(fn)
        stats_at_bldgs.set_index("VDA_id", inplace=True)
        stats_at_bldgs = stats_at_bldgs[~stats_at_bldgs.index.duplicated(keep="first")]
        df_dmg = pd.merge(
            df_dmg, stats_at_bldgs, left_index=True, right_index=True, how="left"
        )

        # ---
        """ each column with observations for:
                [ALL]: all buildings vs. standing / not standing
                [MICRO]: false predictions in xbeach vs. observations.
            """
        # col = "NSI_bldgtype"          # [ALL] H (manufactured) destroyed   | [MICRO] -
        # col = "LC_occupancy_type"     # [ALL] manufactured homes destroyed | [MICRO] -
        # col = "VDA_breakaway_walls"   # [ALL] -                            | [MICRO] breakaway walls result in standing building ***
        # col = "TA_BldgUseTyp"         # [ALL] destroyed mobile homes       | [MICRO] -
        # col="TA_ActYearBuilt_pre1970" # [ALL] before 1970, more destroyed  | [MICRO] before 1970, more destroyed ***
        # col = "TA_EffYearBuilt"       # [ALL] -                            | [MICRO] before 1990, more destroyed
        # col = "FEC_Building_Use"      # [ALL] -                            | [MICRO] -
        # col = "FFE_bldg_diagram"      # [ALL] "8" results in destroyed     | [MICRO] 1a (slab on grade) result in destroyed buildings
        # col = "TA_ShapeSTArea_Sqft"
        col = "max_stat_cumulative_horizontal_impulse"

        # df_dmg = pd.merge(df_xbeach, df_dmg[["removed_vda", "TA_ShapeSTArea_Sqft", "TA_ActYearBuilt"]], left_index=True, right_index=True)

        # df_dmg = df_dmg.dropna(subset=[col])

        # -- four subplots; one for each corner of confusion matrix

        df_true_standing = df_dmg.loc[
            (df_dmg["remove"] == 0) & (df_dmg["removed_vda"] == 0)
        ].index.to_list()
        df_true_destroyd = df_dmg.loc[
            (df_dmg["remove"] == 1) & (df_dmg["removed_vda"] == 1)
        ].index.to_list()
        df_false_standing = df_dmg.loc[
            (df_dmg["remove"] == 0) & (df_dmg["removed_vda"] == 1)
        ].index.to_list()
        df_false_destroyd = df_dmg.loc[
            (df_dmg["remove"] == 1) & (df_dmg["removed_vda"] == 0)
        ].index.to_list()

        df_true_standing = df_dmg.loc[df_true_standing]
        df_true_destroyd = df_dmg.loc[df_true_destroyd]
        df_false_standing = df_dmg.loc[df_false_standing]
        df_false_destroyd = df_dmg.loc[df_false_destroyd]

        fig2, ax = plt.subplots(2, 2, figsize=(8, 6))
        df_true_standing[col].hist(ax=ax[0, 0], grid=False, bins=20)
        df_false_destroyd[col].hist(ax=ax[0, 1], grid=False, bins=20)
        df_false_standing[col].hist(ax=ax[1, 0], grid=False, bins=20)
        df_true_destroyd[col].hist(ax=ax[1, 1], grid=False, bins=20)

        ax[0, 0].text(
            s=len(df_true_standing), x=0.1, y=0.8, transform=ax[0, 0].transAxes
        )
        ax[0, 1].text(
            s=len(df_false_destroyd), x=0.1, y=0.8, transform=ax[0, 1].transAxes
        )
        ax[1, 0].text(
            s=len(df_false_standing), x=0.1, y=0.8, transform=ax[1, 0].transAxes
        )
        ax[1, 1].text(
            s=len(df_true_destroyd), x=0.1, y=0.8, transform=ax[1, 1].transAxes
        )

        plt.tight_layout()
        # self.save_fig(fig2, "confusion-{}" .format(col), dpi=1000)
        plt.show()
