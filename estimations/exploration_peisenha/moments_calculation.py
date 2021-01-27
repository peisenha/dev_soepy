import pandas as pd

def get_moments(df):
    num_periods = df.index.get_level_values("Period").max()

    # Chioce probabilites in each period.
    df_probs_grid = pd.DataFrame(data=0, columns=["Value"], index=pd.MultiIndex.from_product(
        [list(range(num_periods)), ["Home", "Part", "Full"]], names=["Period", "Choice"]))
    df_probs = df.groupby("Period").Choice.value_counts(normalize=True).rename("Value")
    df_probs_grid.update(df_probs)
    moments = list(df_probs_grid.sort_index().values[:, 0])

    return moments
