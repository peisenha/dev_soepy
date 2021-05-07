import os

import pickle as pkl
import pandas as pd

from dev_library import get_weighting_matrix

from moments import get_moments

# We use the unweighted sample for the weighting matrix and then the weighted for the descriptives.
fname_data = os.environ["DATA_DIR"] + "/df-observed-data-unweighted.pkl"
df_obs_unweighted = pd.read_pickle(fname_data)

info = get_weighting_matrix(df_obs_unweighted, get_moments, 500)
pkl.dump(info, open("weighting-matrix.pkl", "wb"))

fname_data = os.environ["DATA_DIR"] + "/df-observed-data-weighted-subsample.pkl"
df_obs_weighted = pd.read_pickle(fname_data)

info = get_moments(df_obs_weighted)
pkl.dump(info, open("observed-moments.pkl", "wb"))
