import sys
import os

import pickle as pkl
import pandas as pd

from dev_library import get_weighting_matrix
from dev_library import df_alignment

# TODO: This import determines the set of moments used for the creation of the weighting matrix
#  and the set of observed moments.
fname_moments = os.environ["PROJECT_DIR"] + "/estimations/exploration_peisenha_drop_high/"
sys.path.insert(0, fname_moments)
from moments import get_moments

fname_data = os.environ["PROJECT_DIR"] + "/resources/soepcore_struct_prep.dta"
df_obs = df_alignment(pd.read_stata(fname_data, convert_categoricals=False))

pkl.dump(get_weighting_matrix(df_obs, get_moments, 500), open("weighting_matrix.pkl", "wb"))
pkl.dump(get_moments(df_obs), open("observed_moments.pkl", "wb"))
