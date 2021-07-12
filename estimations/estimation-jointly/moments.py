import pandas as pd
import numpy as np

LABELS_EDUCATION = ["High", "Medium", "Low"]
LABELS_CHOICE = ["Home", "Part", "Full"]
LABELS_AGE = ["0-2", "3-5", "6-10"]
LABELS_WORK = ["Part", "Full"]
LABELS_CHILD = [False, True]


def get_moments(df):

    df_int = df.copy()

    # We need to add information on whether a child is present.
    try:
        df_int["Child_present"] = (df_int["Number_of_Children"] > 0)
    except:
        df_int["Child_present"] = (df_int["Age_Youngest_Child"] >= 0)

    # For the observed dataset, we have many missing values in our dataset and so we must
    # restrict attention to those that work and make sure we have a numeric type.
    df_working = df_int[df_int["Choice"].isin(LABELS_WORK)]
    df_working = df_working.astype({"Wage_Observed": np.float})

    # We need to add information on the age range of the youngest child and construct some
    # auxiliary variables.
    bins = pd.IntervalIndex.from_tuples([(-0.1, 2.1), (2.9, 5.1), (5.9, 10.1)])
    df_int["Age_Range"] = pd.cut(df_int["Age_Youngest_Child"], bins)
    df_int["Age_Range"].cat.rename_categories(LABELS_AGE, inplace=True)
    num_periods = df_int.index.get_level_values("Period").max()
    moments = []

    # Choice probabilities, differentiating by education, default entry is zero
    entries = [list(range(num_periods)), LABELS_EDUCATION, LABELS_CHOICE]
    conditioning = ["Period", "Education_Level", "Choice"]
    default_entry = 0

    index = pd.MultiIndex.from_product(entries, names=conditioning)
    grid = pd.DataFrame(data=default_entry, columns=["Value"], index=index)

    info = df_int.groupby(conditioning[:2]).Choice.value_counts(normalize=True)
    grid.update(info.rename("Value"))

    # We drop all information on early decisions among the high educated due to data issues.
    index = pd.MultiIndex.from_product([range(5), ["High"], LABELS_CHOICE])
    grid = grid.drop(index)

    moments += grid.sort_index()["Value"].to_list()

    # Choice probabilities, differentiating by age range of youngest child, default entry is zero
    # We restrict attention to the first 20 periods as afterwards the cells get rather thin
    entries = [list(range(20)), LABELS_AGE, LABELS_CHOICE]
    conditioning = ["Period", "Age_Range", "Choice"]
    default_entry = 0

    index = pd.MultiIndex.from_product(entries, names=conditioning)
    grid = pd.DataFrame(data=default_entry, columns=["Value"], index=index)

    info = df_int.groupby(conditioning[:2])["Choice"].value_counts(normalize=True)
    grid.update(info.rename("Value"))

    moments += grid.sort_index()["Value"].to_list()

    # Choice probabilities by presence of child and education level
    entries = [list(range(num_periods)), LABELS_EDUCATION, LABELS_CHILD, LABELS_CHOICE]
    conditioning = ["Period", "Education_Level", "Child_present", "Choice"]
    default_entry = 0

    index = pd.MultiIndex.from_product(entries, names=conditioning)
    grid = pd.DataFrame(data=default_entry, columns=["Value"], index=index)

    info = df_int.groupby(conditioning[:3]).Choice.value_counts(normalize=True)
    grid.update(info.rename("Value"))

    moments += grid.sort_index()["Value"].to_list()

    # Average wages, differentiating by education, default entry is average wage in sample
    entries = [list(range(num_periods)), LABELS_EDUCATION, LABELS_WORK]
    conditioning = ["Period", "Education_Level", "Choice"]
    default_entry = df_int["Wage_Observed"].mean()

    index = pd.MultiIndex.from_product(entries, names=conditioning)
    grid = pd.DataFrame(data=default_entry, columns=["Value"], index=index)

    info = df_working.groupby(conditioning)["Wage_Observed"].mean()
    grid.update(info.rename("Value"))

    moments += grid.sort_index()["Value"].to_list()

    # Average wages, differentiating by education and experience, default entry is average wage
    # in sample.
    default_entry = df_working["Wage_Observed"].mean()

    for choice in LABELS_WORK:
        exp_label = f"Experience_{choice}_Time"

        conditioning = ["Choice", "Education_Level", exp_label]
        entries = [[choice], LABELS_EDUCATION, range(20)]

        index = pd.MultiIndex.from_product(entries, names=conditioning)
        grid = pd.DataFrame(data=default_entry, columns=["Value"], index=index)

        info = df_working.groupby(conditioning)["Wage_Observed"].mean()
        grid.update(info.rename("Value"))

        # We drop all information on early decisions among the high educated due to data issues.
        index = pd.MultiIndex.from_product([[choice], ["High"], range(5)])
        grid = grid.drop(index)

        moments += grid.sort_index()["Value"].to_list()

    # Distribution of wages, default entry is average wage in sample.
    default_entry = df_working["Wage_Observed"].mean()

    quantiles = [0.1, 0.25, 0.50, 0.75, 0.9]
    conditioning = ["Period", "Choice", "Quantile"]
    entries = [list(range(num_periods)), LABELS_WORK, quantiles]

    index = pd.MultiIndex.from_product(entries, names=conditioning)
    grid = pd.DataFrame(data=default_entry, columns=["Value"], index=index)

    info = df_working.groupby(conditioning[:2])["Wage_Observed"].quantile(quantiles)
    grid.update(info.rename("Value"))

    moments += grid.sort_index()["Value"].to_list()

    # TODO: The next line rules out any issues in the variance calculation du to the duplicated
    # entries in the observed dataset for the weighting.
    df_working.drop_duplicates(inplace=True)

    # Variance of wages by work status, overall, default entry is variance of wage in sample
    default_entry = df_working["Wage_Observed"].var()

    conditioning = ["Period", "Choice"]
    entries = [list(range(num_periods)), LABELS_WORK]

    index = pd.MultiIndex.from_product(entries, names=conditioning)
    grid = pd.DataFrame(data=default_entry, columns=["Value"], index=index)

    info = df_working.groupby(conditioning)["Wage_Observed"].var()
    grid.update(info.rename("Value"))

    moments += grid.sort_index()["Value"].to_list()

    return moments
