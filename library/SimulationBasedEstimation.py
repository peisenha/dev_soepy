import yaml
import os

import pandas as pd
import numpy as np

import soepy

HUGE_INT = 100000000000


class SimulationBasedEstimationCls:
    """This class facilitates estimation of the free parameter vector
    in a life-cycle model of labor supply based on the soepy package
    and smm_estimagic."""

    def __init__(
        self,
        params,
        model_spec_init_file_name,
        moments_obs,
        weighting_matrix,
        get_moments,
        logging_dir=os.getcwd(),
        log_file_name_extension="",
        max_evals=HUGE_INT,
    ):

        self.params = params
        self.model_spec_init_file_name = model_spec_init_file_name
        self.moments_obs = moments_obs
        self.weighting_matrix = weighting_matrix
        self.get_moments = get_moments
        self.max_evals = max_evals
        self.log_file_name_extension = log_file_name_extension
        self.logging_dir = logging_dir
        self.num_evals = 0
        self.fval = None

        self._calculate_criterion_func_value(self.params)

    def get_objective(self, params_cand):

        self.params = params_cand
        self.params.drop(columns=["_fixed"], inplace=True, errors="ignore")

        # Obtain criterion function value
        fval, stats_obs, stats_sim = self._calculate_criterion_func_value(params_cand)

        # We need to prepare logging in a separate directory. This needs to be done here so the
        # parameterization at the start can be saved there right away.
        if not os.path.exists(self.logging_dir):
            os.makedirs(self.logging_dir)

        # Save params and function value as pickle object.
        is_start = self.fval is None

        if is_start:
            data = {"current": fval, "start": fval, "step": fval}
            self.fval = pd.DataFrame(
                data, columns=["current", "start", "step"], index=[0]
            )
            self.params.to_pickle(self.logging_dir + "/step.soepy.pkl")
        else:
            is_step = self.fval["step"].iloc[-1] > fval
            step = self.fval["step"].iloc[-1]
            start = self.fval["start"].loc[0]

            if is_step:
                data = {"current": fval, "start": start, "step": fval}
                self.params.to_pickle(self.logging_dir + "/step.soepy.pkl")
            else:
                data = {"current": fval, "start": start, "step": step}

            self.fval = self.fval.append(data, ignore_index=True)

        self._logging_smm(stats_obs, stats_sim, fval)

        self.num_evals = self.num_evals + 1
        if self.num_evals >= self.max_evals:
            raise RuntimeError("maximum number of evaluations reached")

        return fval

    def _calculate_criterion_func_value(self, params_cand):

        self.params = params_cand

        # Extract elements from configuration file.
        benefits_base = float(params_cand.loc["benefits_base", "value"].values[0])
        delta = float(params_cand.loc["delta", "value"].values[0])
        mu = float(params_cand.loc["mu", "value"].values[0])

        model_spec_init_dict = yaml.load(open(self.model_spec_init_file_name), Loader=yaml.Loader)
        model_spec_init_dict["TAXES_TRANSFERS"]["benefits_base"] = benefits_base
        model_spec_init_dict["CONSTANTS"]["delta"] = delta
        model_spec_init_dict["CONSTANTS"]["mu"] = mu

        fname_modified = "resources/model_spec_init.modified.yml"
        yaml.dump(model_spec_init_dict, open(fname_modified, "w"))

        # Generate simulated data set
        data_frame_sim = soepy.simulate(self.params, fname_modified)
        data_frame_sim = df_alignment(data_frame_sim)

        # Calculate simulated moments
        moments_sim = self.get_moments(data_frame_sim)

        # Construct criterion value
        stats_dif = np.array(self.moments_obs) - np.array(moments_sim)

        fval = float(np.dot(np.dot(stats_dif, self.weighting_matrix), stats_dif))

        return fval, self.moments_obs,  moments_sim

    def _logging_smm(self, stats_obs, stats_sim, fval):
        """This method contains logging capabilities
        that are just relevant for the SMM routine."""
        fname = (
            self.logging_dir
            + "/monitoring.smm_estimagic."
            + self.log_file_name_extension
            + ".info"
        )
        fname2 = (
            self.logging_dir
            + "/monitoring_compact.smm_estimagic."
            + self.log_file_name_extension
            + ".info"
        )

        if self.num_evals == 0 and os.path.exists(fname):
            os.unlink(fname)
        if self.num_evals == 0 and os.path.exists(fname2):
            os.unlink(fname2)

        with open(fname, "a+") as outfile:
            fmt_ = "\n\n{:>8}{:>15}\n\n"
            outfile.write(fmt_.format("EVALUATION", self.num_evals))
            fmt_ = "\n\n{:>8}{:>15}\n\n"
            outfile.write(fmt_.format("fval", round(fval, 5)))
            for x in self.params.index:
                info = [x[0], x[1], round(self.params.loc[x, "value"], 5)]
                fmt_ = "{:>8}" + "{:>15}" * 2 + "\n\n"
                outfile.write(fmt_.format(*info))

            fmt_ = "{:>8}" + "{:>15}" * 4 + "\n\n"
            info = ["Moment", "Observed", "Simulated", "Difference", "Weight"]
            outfile.write(fmt_.format(*info))
            for x in enumerate(stats_obs):
                stat_obs, stat_sim = stats_obs[x[0]], stats_sim[x[0]]
                info = [
                    x[0],
                    stat_obs,
                    stat_sim,
                    abs(stat_obs - stat_sim),
                    self.weighting_matrix[x[0], x[0]],
                ]

                fmt_ = "{:>8}" + "{:15.5f}" * 4 + "\n"
                outfile.write(fmt_.format(*info))

        with open(fname2, "a+") as outfile:
            fmt_ = "\n\n{:>8}{:>15}\n\n"
            outfile.write(fmt_.format("EVALUATION", self.num_evals))
            fmt_ = "\n\n{:>8}{:>15}\n\n"
            outfile.write(fmt_.format("fval", round(fval, 5)))
            for x in self.params.index:
                info = [x[0], x[1], round(self.params.loc[x, "value"], 5)]
                fmt_ = "{:>8}" + "{:>15}" * 2 + "\n\n"
                outfile.write(fmt_.format(*info))

def df_alignment(df):
    df_int = df.copy()
    rename = dict()
    rename["Choice"] = {0: "Home", 1: "Part", 2: "Full"}
    rename["Education_Level"] = {0: "Low", 1: "Medium", 2: "High"}

    df_int.replace(rename, inplace=True)

    df_int.set_index(["Identifier", "Period"], inplace=True)


    return df_int