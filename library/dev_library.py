"""This module contains functions to calculate the moments based on the simulated
data."""
import os

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


LABELS_EDUCATION = ["High", "Medium", "Low"]
LABELS_CHOICE = ["Home", "Part", "Full"]
LABELS_WORK = ["Part", "Full"]

"""This module contains functions to calculate the moments based on the simulated
data."""


def get_criterion_function(moments_obs, moments_sim, weighting_matrix):
    stats_dif = np.array(moments_obs) - np.array(moments_sim)
    return float(np.dot(np.dot(stats_dif, weighting_matrix), stats_dif))


def get_weighting_matrix(data_frame, get_moments, num_samples):
    """Calculates the weighting matrix based on the
    moments of the observed data"""

    data_frame_intern = data_frame.copy()

    moments_sample = []

    identifiers = data_frame.index.get_level_values("Identifier").unique().to_list()

    # Collect n samples of moments
    for k in range(num_samples):
        print(k)
        identifiers = np.random.choice(identifiers, replace=True, size=len(identifiers))
        df_boot = data_frame_intern.loc[(identifiers, slice(None)), :]
        moments_sample.append(get_moments(df_boot))

    # Calculate sample variances for each moment
    moments_var = np.array(moments_sample).var(axis=0)

    # Handling of zero variances
    is_zero = moments_var <= 1e-10
    moments_var[is_zero] = 0.1

    # Construct weighting matrix
    weighting_matrix = np.diag(moments_var ** (-1))

    return weighting_matrix


def df_alignment(df, is_obs=False):
    df_int = df.copy()
    rename = dict()
    rename["Choice"] = {0: "Home", 1: "Part", 2: "Full"}
    rename["Education_Level"] = {0: "Low", 1: "Medium", 2: "High"}
    df_int.replace(rename, inplace=True)

    df_int.set_index(["Identifier", "Period"], inplace=True)
    df_int.sort_index(inplace=True)

    if is_obs:
        num_persons = df_int.index.get_level_values("Identifier").nunique()
        max_period = df_int.index.get_level_values("Period").max()
        columns = df_int.columns
        df_int.index.set_levels(range(num_persons), level="Identifier", inplace=True)

        index = pd.MultiIndex.from_product(
            [range(num_persons), range(max_period + 1)], names=["Identifier", "Period"]
        )
        df_grid = pd.DataFrame(data=None, columns=columns, index=index)
        df_grid.update(df_int)
    else:
        df_grid = df_int

    return df_grid

def plot_children_choices(df_sim, df_obs):
    
    df_obs["Child_present"] = (df_obs["Number_of_Children"] > 0)
    df_sim["Child_present"] = (df_sim["Age_Youngest_Child"] >= 0)
    
    num_periods = df_obs.index.get_level_values("Period").max()

    LABELS_EDUCATION = ["High", "Medium", "Low"]
    LABELS_CHOICE = ["Home", "Part", "Full"]
    LABELS_CHILD = [False, True]
    
    def get_info(df_int):

        entries = [list(range(num_periods)), LABELS_EDUCATION, LABELS_CHILD, LABELS_CHOICE]
        conditioning = ["Period", "Education_Level", "Child_present", "Choice"]
        default_entry = 0

        index = pd.MultiIndex.from_product(entries, names=conditioning)
        grid = pd.DataFrame(data=default_entry, columns=["Value"], index=index)

        info = df_int.groupby(conditioning[:3]).Choice.value_counts(normalize=True)
        grid.update(info.rename("Value"))

        return grid

    info_obs = get_info(df_obs)
    info_sim = get_info(df_sim)
    
    
    
    
    x_values = range(num_periods)

    for edu_level in ["Low", "Medium", "High"]:

        fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=(16, 8))

        for ax, choice in [(ax1, "Full"), (ax2, "Home"), (ax3, "Part")]:        
            ax.set_prop_cycle(None)

            y_sim = info_sim.loc[slice(None), edu_level, True, choice]
            ax.plot(x_values, y_sim, label=f"Simulated")

            y_obs = info_obs.loc[slice(None), edu_level, True, choice]
            ax.plot(x_values, y_obs, label=f"Observed")


            ax.set_title(f"{edu_level}, {choice}, with children")
            ax.set_ylim([0, 1])    
            ax.legend()
            

def plot_basics_choices(df_obs, df_sim):

    for choice in ["Full", "Home", "Part"]:
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(nrows=1, ncols=4, figsize=(16, 8))

        for edu_level, ax in [
            ("High", ax1),
            ("Medium", ax2),
            ("Low", ax3),
            ("All", ax4),
        ]:

            if edu_level != "All":
                df_sim_subset = df_sim[df_sim["Education_Level"] == edu_level]
                df_obs_subset = df_obs[df_obs["Education_Level"] == edu_level]
            else:
                df_sim_subset = df_sim
                df_obs_subset = df_obs

            y_sim = (
                df_sim_subset.groupby("Period")
                .Choice.value_counts(normalize=True).sort_index()
                .loc[(slice(None), choice)]
            )
            y_obs = (
                df_obs_subset.groupby("Period")
                .Choice.value_counts(normalize=True).sort_index()
                .loc[(slice(None), choice)]
            )

            x = df_sim.index.get_level_values("Period").unique()
            try:
                ax.plot(x, y_sim, label="Simulated")
                ax.plot(x, y_obs, label="Observed")
                ax.legend()
                ax.set_ylim([0, 1])
                ax.set_title(f"{choice}, {edu_level}")
            except ValueError:
                pass

def plot_basics_wages(df_obs, df_sim, std=False):

    for work_level in ["Full", "Part"]:

        df_sim_work_level = df_sim[df_sim["Choice"] == work_level]
        df_obs_work_level = df_obs[df_obs["Choice"] == work_level]

        df_obs_work_level = df_obs_work_level.astype({"Wage_Observed": np.float})

        fig, (ax1, ax2, ax3, ax4) = plt.subplots(nrows=1, ncols=4, figsize=(16, 8))

        for edu_level, ax in [
            ("High", ax1),
            ("Medium", ax2),
            ("Low", ax3),
            ("All", ax4),
        ]:

            if edu_level != "All":
                df_sim_subset = df_sim_work_level[
                    df_sim_work_level["Education_Level"] == edu_level
                ]
                df_obs_subset = df_obs_work_level[
                    df_obs_work_level["Education_Level"] == edu_level
                ]
            else:
                df_sim_subset = df_sim_work_level
                df_obs_subset = df_obs_work_level

            y_sim = df_sim_subset.groupby("Period")["Wage_Observed"].mean()
            y_obs = df_obs_subset.groupby("Period")["Wage_Observed"].mean()

            x = df_sim.index.get_level_values("Period").unique()

            ax.plot(x, y_sim, label="Simulated")
            ax.plot(x, y_obs, label="Observed")
            ax.legend()
            ax.set_title(f"Mean, {work_level}, {edu_level}")

        if not std:
            continue

        fig, (ax1, ax2, ax3, ax4) = plt.subplots(nrows=1, ncols=4, figsize=(16, 8))

        for edu_level, ax in [
            ("High", ax1),
            ("Medium", ax2),
            ("Low", ax3),
            ("All", ax4),
        ]:

            if edu_level != "All":
                df_sim_subset = df_sim_work_level[
                    df_sim_work_level["Education_Level"] == edu_level
                ]
                df_obs_subset = df_obs_work_level[
                    df_obs_work_level["Education_Level"] == edu_level
                ]
            else:
                df_sim_subset = df_sim_work_level
                df_obs_subset = df_obs_work_level

            y_sim = df_sim_subset.groupby("Period")["Wage_Observed"].std()
            y_obs = df_obs_subset.groupby("Period")["Wage_Observed"].std()

            x = df_sim.index.get_level_values("Period").unique()

            ax.plot(x, y_sim, label="Simulated")
            ax.plot(x, y_obs, label="Observed")
            ax.legend()
            ax.set_title(f"Standard deviation, {work_level}, {edu_level}")


def get_observed_moments(get_moments):

    fname = os.environ["PROJECT_DIR"] + "/resources/soepcore_struct_prep.dta"
    df_obs = pd.read_stata(fname, convert_categoricals=False)
    df_obs = df_alignment(df_obs)
    return get_moments(df_obs)
