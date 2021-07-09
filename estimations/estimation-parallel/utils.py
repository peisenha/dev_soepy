import pickle as pkl
import sys
import os

import pandas as pd

def common_setup():
    cwd = os.getcwd()

    # We need to set up our criterion function.
    os.chdir(os.environ["PROJECT_DIR"] + "/estimations/estimation-parallel/resources")
    sys.path.insert(0, os.environ["PROJECT_DIR"] + "/estimations/estimation-parallel/resources")
    from moments import get_moments

    params_start = pd.read_pickle("start.soepy.pkl")
    params_start["fixed"] = True

    adapter_kwargs = dict()
    adapter_kwargs["weighting_matrix"] = pkl.load(open("weighting-matrix.pkl", "rb"))
    adapter_kwargs["moments_obs"] = pkl.load(open("observed-moments.pkl", "rb"))
    adapter_kwargs["model_spec_init_file_name"] = os.getcwd() + "/model_spec_init.yml"
    adapter_kwargs["get_moments"] = get_moments
    os.chdir(cwd)

    opt_kwargs = dict()
    opt_kwargs["scaling_within_bounds"] = True
    opt_kwargs["seek_global_minimum"] = True
    opt_kwargs["objfun_has_noise"] = True

    return params_start, adapter_kwargs, opt_kwargs
