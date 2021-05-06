import pandas as pd

from dev_library import *

fname = os.environ["PROJECT_DIR"] + "/resources/soepcore_struct_prep_revised.dta"
df_obs = pd.read_stata(fname, convert_categoricals = False)
df_obs = df_alignment(df_obs, is_obs=True)
df_obs.dropna(axis=0, how="all", inplace=True)

for period in df_obs.index.get_level_values(1).unique():

    df_subset = df_obs.loc[(slice(None), period), :].sample(3)

    rslt = list()
    counter = 0
    for idx, row in df_subset.iterrows():
        counter += 1
        weight = int(row["Person_Weight"])
        if weight > 0:
            rslt += [pd.concat([pd.DataFrame(row).T]* weight)]

    df_extended = pd.concat(rslt)
    df_extended.index.names = ["Identifier", "Period"]

    df_extended.to_pickle(f"df_obs_weighted_{period}.pkl")