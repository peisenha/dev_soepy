import pandas as pd
import pickle as pkl

from dev_library import get_weighting_matrix
from dev_library import df_alignment

df_obs = pd.read_stata("../resources/soepcore_struct_prep.dta", convert_categoricals=False)
df_obs = df_alignment(df_obs)
pkl.dump(get_weighting_matrix(df_obs, 500), open("weighting_matrix.pkl", "wb"))
