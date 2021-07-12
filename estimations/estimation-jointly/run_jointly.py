#!usr/bin/env python
from functools import partial
import pickle as pkl
import sys

import pybobyqa as pybob
import pandas as pd
import numpy as np

from SimulationBasedEstimation import SimulationBasedEstimationCls
from pybobyqa_auxiliary import prepare_optimizer_interface
from pybobyqa_auxiliary import wrapper_numpy
from moments import get_moments

# We need to set up our criterion function.
params_start = pd.read_pickle("start.soepy.pkl")

adapter_kwargs = dict()
adapter_kwargs["weighting_matrix"] = pkl.load(open("resources/weighting-matrix.pkl", "rb"))
adapter_kwargs["model_spec_init_file_name"] = "resources/model_spec_init.yml"
adapter_kwargs["moments_obs"] = pkl.load(open("resources/observed-moments.pkl", "rb"))
adapter_kwargs["get_moments"] = get_moments
adapter_kwargs["params"] = params_start


# Setup 
adapter_smm = SimulationBasedEstimationCls(**adapter_kwargs)
np.testing.assert_almost_equal(adapter_smm.fval, 46079.91940236735)

# Estimation
opt_kwargs = dict()
opt_kwargs["scaling_within_bounds"] = True
opt_kwargs["seek_global_minimum"] = True
opt_kwargs["objfun_has_noise"] = True
opt_kwargs["maxfun"] = 100000

x0, bounds = prepare_optimizer_interface(params_start)
p_wrapper_numpy = partial(wrapper_numpy, params_start, adapter_smm)
rslt = pybob.solve(p_wrapper_numpy, x0, bounds=bounds, **opt_kwargs)
