#!usr/bin/env python
from functools import partial
import sys

import pybobyqa as pybob
import numpy as np

from SimulationBasedEstimation import SimulationBasedEstimationCls
from pybobyqa_auxiliary import prepare_optimizer_interface
from pybobyqa_auxiliary import wrapper_numpy

sys.path.insert(0, "../")
from utils import common_setup

params_start, adapter_kwargs, opt_kwargs = common_setup()

params_start.loc[("const_wage_eq", "gamma_0s1"), "fixed"] = False
params_start.loc[("exp_returns", "gamma_1s1"), "fixed"] = False

params_start.loc[("disutil_work", "no_kids_f_educ_low"),  "fixed"] = False
params_start.loc[("disutil_work", "no_kids_p_educ_low"),  "fixed"] = False
params_start.loc[("disutil_work", "yes_kids_f_educ_low"), "fixed"] = False
params_start.loc[("disutil_work", "yes_kids_p_educ_low"), "fixed"] = False

params_start.loc[("disutil_work", "no_kids_f_educ_low"), ["upper", "lower"]] = [1.25, 0.75]
params_start.loc[("disutil_work", "no_kids_p_educ_low"), ["upper", "lower"]] = [-0.0, -1.00]
params_start.loc[("disutil_work", "yes_kids_p_educ_low"), ["upper", "lower"]] = [-0.5, -1.00]
params_start.to_pickle("start_updated.soepy.pkl")

# Setup 
adapter_kwargs["params"] = params_start
adapter_smm = SimulationBasedEstimationCls(**adapter_kwargs)
np.testing.assert_almost_equal(adapter_smm.fval, 128384.7170351552)

# Estimation
opt_kwargs["maxfun"] = 100000

x0, bounds = prepare_optimizer_interface(params_start)
p_wrapper_numpy = partial(wrapper_numpy, params_start, adapter_smm)
rslt = pybob.solve(p_wrapper_numpy, x0, bounds=bounds, **opt_kwargs)
