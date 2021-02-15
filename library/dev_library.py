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


def get_weighting_matrix(data_frame, get_moments, num_samples):
    """Calculates the weighting matrix based on the
    moments of the observed data"""

    data_frame_intern = data_frame.copy()

    moments_sample = []

    identifiers = data_frame.index.get_level_values("Identifier").unique().to_list()

    # Collect n samples of moments
    for k in range(num_samples):
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


def transitions_out_to_in(data_subset, num_periods):
    counts_list = []
    for period in np.arange(1, num_periods):
        # get period IDs:
        period_employed_ids = data_subset[
            (data_subset["Period"] == period) & (data_subset["Choice"] != 0)
        ]["Identifier"].to_list()
        transition_ids = data_subset[
            (data_subset["Period"] == period - 1)
            & (data_subset["Identifier"].isin(period_employed_ids))
            & (data_subset["Choice"] == 0)
        ]["Identifier"].to_list()
        period_counts = (
            data_subset[
                (data_subset["Period"] == period)
                & (data_subset["Identifier"].isin(transition_ids))
            ]["Identifier"].count()
            / data_subset[(data_subset["Period"] == period)]["Identifier"].count()
        )
        counts_list += [period_counts]
    avg = np.mean(counts_list)
    return avg


def transitions_out_to_in_mothers(data_subset, num_periods):
    counts_list = []
    for period in np.arange(1, min(28, num_periods)):
        # get period IDs:
        period_employed_ids = data_subset[
            (data_subset["Period"] == period) & (data_subset["Choice"] != 0)
        ]["Identifier"].to_list()
        transition_ids = data_subset[
            (data_subset["Period"] == period - 1)
            & (data_subset["Identifier"].isin(period_employed_ids))
            & (data_subset["Choice"] == 0)
        ]["Identifier"].to_list()
        period_counts = (
            data_subset[
                (data_subset["Period"] == period)
                & (data_subset["Identifier"].isin(transition_ids))
            ]["Identifier"].count()
            / data_subset[(data_subset["Period"] == period)]["Identifier"].count()
        )
        counts_list += [period_counts]
    avg = np.mean(counts_list)
    return avg


def transitions_in_to_out(data_subset, num_periods):
    counts_list = []
    for period in np.arange(1, num_periods):
        # get period IDs:
        period_unemployed_ids = data_subset[
            (data_subset["Period"] == period) & (data_subset["Choice"] == 0)
        ]["Identifier"].to_list()
        transition_ids = data_subset[
            (data_subset["Period"] == period - 1)
            & (data_subset["Identifier"].isin(period_unemployed_ids))
            & (data_subset["Choice"] != 0)
        ]["Identifier"].to_list()
        period_counts = (
            data_subset[
                (data_subset["Period"] == period)
                & (data_subset["Identifier"].isin(transition_ids))
            ]["Identifier"].count()
            / data_subset[(data_subset["Period"] == period)]["Identifier"].count()
        )
        counts_list += [period_counts]
    avg = np.mean(counts_list)
    return avg


def transitions_in_to_out_mothers(data_subset, num_periods):
    counts_list = []
    for period in np.arange(1, min(28, num_periods)):
        # get period IDs:
        period_unemployed_ids = data_subset[
            (data_subset["Period"] == period) & (data_subset["Choice"] == 0)
        ]["Identifier"].to_list()
        transition_ids = data_subset[
            (data_subset["Period"] == period - 1)
            & (data_subset["Identifier"].isin(period_unemployed_ids))
            & (data_subset["Choice"] != 0)
        ]["Identifier"].to_list()
        period_counts = (
            data_subset[
                (data_subset["Period"] == period)
                & (data_subset["Identifier"].isin(transition_ids))
            ]["Identifier"].count()
            / data_subset[(data_subset["Period"] == period)]["Identifier"].count()
        )
        counts_list += [period_counts]
    avg = np.mean(counts_list)
    return avg


def transitions_in_to_out_deciles(data, decile, num_periods):
    counts_list = []
    for period in np.arange(1, num_periods):
        # get period IDs:
        period_unemployed_ids = data[
            (data["Period"] == period) & (data["Choice"] == 0)
        ]["Identifier"].to_list()
        transition_ids = data[
            (data["Period"] == period - 1)
            & (data["Identifier"].isin(period_unemployed_ids))
            & (data["Wage_Observed"] < data["Wage_Observed"].quantile(decile))
        ]["Identifier"].to_list()
        period_counts = (
            data[
                (data["Period"] == period) & (data["Identifier"].isin(transition_ids))
            ]["Identifier"].count()
            / data[(data["Period"] == period)]["Identifier"].count()
        )
        counts_list += [period_counts]
    avg = np.mean(counts_list)

    return avg


def df_alignment(df):
    df_int = df.copy()
    rename = dict()
    rename["Choice"] = {0: "Home", 1: "Part", 2: "Full"}
    rename["Education_Level"] = {0: "Low", 1: "Medium", 2: "High"}

    df_int.replace(rename, inplace=True)

    df_int.set_index(["Identifier", "Period"], inplace=True)


    return df_int


def plot_basics_choices(df_obs, df_sim):

    for choice in ["Full", "Part", "Home"]:
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(nrows=1, ncols=4, figsize=(16, 8))

        for edu_level, ax in [("High", ax1), ("Medium", ax2), ("Low", ax3), ("All", ax4)]:

            if edu_level != "All":
                df_sim_subset = df_sim[df_sim["Education_Level"] == edu_level]
                df_obs_subset = df_obs[df_obs["Education_Level"] == edu_level]
            else:
                df_sim_subset = df_sim
                df_obs_subset = df_obs

            y_sim = df_sim_subset.groupby("Period").Choice.value_counts(normalize=True).loc[(slice(None), choice)]
            y_obs = df_obs_subset.groupby("Period").Choice.value_counts(normalize=True).loc[(slice(None), choice)]

            x = df_sim.index.get_level_values("Period").unique()

            ax.plot(x, y_sim, label="Simulated")
            ax.plot(x, y_obs, label="Observed")
            ax.legend()
            ax.set_ylim([0, 1])
            ax.set_title(f"{choice}, {edu_level}")

def plot_basics_wages(df_obs, df_sim, std=False):
    
    
    for work_level in ["Full", "Part"]:

        df_sim_work_level = df_sim[df_sim["Choice"] == work_level]
        df_obs_work_level = df_obs[df_obs["Choice"] == work_level]
            
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(nrows=1, ncols=4, figsize=(16, 8))

        for edu_level, ax in [("High", ax1), ("Medium", ax2), ("Low", ax3), ("All", ax4)]:

            if edu_level != "All":
                df_sim_subset = df_sim_work_level[df_sim_work_level["Education_Level"] == edu_level]
                df_obs_subset = df_obs_work_level[df_obs_work_level["Education_Level"] == edu_level]
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

        for edu_level, ax in [("High", ax1), ("Medium", ax2), ("Low", ax3), ("All", ax4)]:

            if edu_level != "All":
                df_sim_subset = df_sim_work_level[df_sim_work_level["Education_Level"] == edu_level]
                df_obs_subset = df_obs_work_level[df_obs_work_level["Education_Level"] == edu_level]
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
