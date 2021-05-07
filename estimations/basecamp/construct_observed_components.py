import os

import pickle as pkl
import pandas as pd

from dev_library import get_weighting_matrix

from moments import get_moments

fname_data = os.environ["DATA_DIR"] + "/df-observed-data-unweighted.pkl"
df_obs_weighted = pd.read_pickle(fname_data)

info = get_weighting_matrix(df_obs_weighted, get_moments, 500)
pkl.dump(info, open("weighting-matrix.pkl", "wb"))

info = get_moments(df_obs_weighted)
pkl.dump(info, open("observed-moments.pkl", "wb"))
