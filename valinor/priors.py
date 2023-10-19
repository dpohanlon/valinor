import pandas as pd

import numpy as np

from typing import Dict, List, Tuple, Any


def calculateOverdispersion(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate overdispersion in the given DataFrame.

    Args:
        df (pd.DataFrame): The input DataFrame.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Mean and standard deviation of overdispersion.
    """

    reps = (
        df.groupby(["GuidePair", "cell_line_index"])
        .agg(
            mean=("value", np.mean),
            std=("value", np.std),
            cell_line_index=("cell_line_index", "first"),
        )
        .reset_index(drop=True)
    )

    reps["var"] = reps["std"] ** 2
    reps["od"] = reps["var"] / reps["mean"]

    repsC = reps.groupby(["cell_line_index"]).agg(
        mean=("od", np.median), std=("od", np.std)
    )
    repsC = repsC.sort_values("cell_line_index")

    return repsC["mean"].values, repsC["std"].values


def defaultPriors():
    prior_params = {}

    prior_params["guide_eff_mean"] = (0.9, 0.1)
    prior_params["guide_eff_std"] = (0.1, 0.1)

    prior_params["gene_ko_growth"] = (0.0, 0.1)

    prior_params["cell_line_growth"] = (0.0, 0.1)

    prior_params["mv_mean_scale"] = 1.0
    prior_params["mv_std_scale"] = 1.0

    prior_params["pair_eff_mean"] = (0.9, 0.1)
    prior_params["pair_eff_std"] = (0.1, 0.1)

    prior_params["pair_growth"] = (0.0, 0.1)

    return prior_params
