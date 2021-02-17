import pandas as pd

LABELS_EDUCATION = ["High", "Medium", "Low"]
LABELS_CHOICE = ["Home", "Part", "Full"]
LABELS_WORK = ["Part", "Full"]


def get_moments(df):

    df_int = df.copy()

    num_periods = df_int.index.get_level_values("Period").max()

    # Choice probabilities, differentiating by education, default entry is zero
    entries = [list(range(num_periods)), LABELS_EDUCATION, LABELS_CHOICE]
    conditioning = ["Period", "Education_Level", "Choice"]
    default_entry = 0

    index = pd.MultiIndex.from_product(entries, names=conditioning)
    df_probs_grid = pd.DataFrame(data=default_entry, columns=["Value"], index=index)

    df_probs = df_int.groupby(conditioning[:2]).Choice.value_counts(normalize=True).rename("Value")
    df_probs_grid.update(df_probs)

    # We drop all information on early decisions among the high educated due to data issues.
    index = pd.MultiIndex.from_product([range(5), ["High"], LABELS_CHOICE])
    df_probs_grid = df_probs_grid.drop(index)

    moments = list(df_probs_grid.sort_index().values.flatten())

    # Average wages, differentiating by education, default entry is average wage in sample
    entries = [list(range(num_periods)), LABELS_EDUCATION, LABELS_WORK]
    conditioning = ["Period", "Education_Level", "Choice"]
    default_entry = df_int["Wage_Observed"].mean()

    index = pd.MultiIndex.from_product(entries, names=conditioning)
    df_wages_mean_grid = pd.DataFrame(data=default_entry, columns=["Value"], index=index)

    df_sim_working = df_int[df_int["Choice"].isin(LABELS_WORK)]
    df_wages_mean = df_sim_working.groupby(conditioning)["Wage_Observed"].mean().rename("Value")
    df_wages_mean_grid.update(df_wages_mean)

    moments += list(df_wages_mean_grid.sort_index().values.flatten())

    # Variance of wages by work status, overall, default entry is variance of wage in sample
    default_entry = df_int["Wage_Observed"].var()

    index = ["Full", "Part"]
    df_wages_var_grid = pd.DataFrame(data=default_entry, columns=["Value"], index=index)

    df_sim_working = df_int[df_int["Choice"].isin(LABELS_WORK)]
    df_wages_var = df_sim_working.groupby(["Choice"])["Wage_Observed"].var().rename("Value")
    df_wages_var_grid.update(df_wages_var)

    moments += list(df_wages_var_grid.sort_index().values.flatten())

    # Persistence in choices
    df_int.loc[:, "Choice_Lagged"] = df_int.groupby("Identifier").shift(1)[["Choice"]]
    rslt = pd.crosstab(df_int["Choice"], df_int["Choice_Lagged"], normalize="index")

    moments += list(rslt.sort_index().values.flatten())

    return moments
