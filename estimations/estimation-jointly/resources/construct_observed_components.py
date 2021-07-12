import os

import pickle as pkl
import pandas as pd

from dev_library import get_weighting_matrix

from moments import get_moments

fname_data = "../../../../data-exchange-sciebo/df-observed-data-weighted-subsample.pkl"
df_obs = pd.read_pickle(fname_data).groupby("Period").sample(1000)

info = get_weighting_matrix(df_obs, get_moments, 500)
pkl.dump(info, open("weighting-matrix.pkl", "wb"))

info = get_moments(df_obs)
pkl.dump(info, open("observed-moments.pkl", "wb"))
