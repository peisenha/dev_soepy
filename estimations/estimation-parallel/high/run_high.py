#!usr/bin/env python
from functools import partial
import sys

import pybobyqa as pybob
import numpy as np

from SimulationBasedEstimation import SimulationBasedEstimationCls
from pybobyqa_auxiliary import prepare_optimizer_interface
from pybobyqa_auxiliary import wrapper_numpy

# Modifications
sys.path.insert(0, "../")
from utils import common_setup

params_start, adapter_kwargs, opt_kwargs = common_setup()

params_start.loc[("disutil_work", "no_kids_f_educ_high"), "fixed"] = False 
params_start.loc[("disutil_work", "no_kids_p_educ_high"), "fixed"] = False 
params_start.loc[("disutil_work", "yes_kids_f_educ_high"), "fixed"] = False
params_start.loc[("disutil_work", "yes_kids_p_educ_high"), "fixed"] = False

params_start.loc[("const_wage_eq", "gamma_0s3"), "fixed"] = False
params_start.loc[("exp_returns", "gamma_1s3"), "fixed"] = False

params_start.loc[("const_wage_eq", "gamma_0s3"), ["lower", "upper"]] = [2.0, 2.5]

params_start.to_pickle("start_updated.soepy.pkl")

# Setup 
adapter_kwargs["params"] = params_start
adapter_smm = SimulationBasedEstimationCls(**adapter_kwargs)
np.testing.assert_almost_equal(adapter_smm.fval, 128384.7170351552)

# Estimation
opt_kwargs["maxfun"] = 10000

x0, bounds = prepare_optimizer_interface(params_start)
p_wrapper_numpy = partial(wrapper_numpy, params_start, adapter_smm)
rslt = pybob.solve(p_wrapper_numpy, x0, bounds=bounds, **opt_kwargs)
