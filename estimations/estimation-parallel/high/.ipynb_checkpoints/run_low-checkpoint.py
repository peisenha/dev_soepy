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

cols = ["value", "fixed"]
params_start.loc[("disutil_work", "no_kids_f_educ_low"), cols] = [+0.90, False]   
params_start.loc[("disutil_work", "no_kids_p_educ_low"), cols] = [-0.75, False]   
params_start.loc[("disutil_work", "yes_kids_f_educ_low"), cols] = [+1.00, False]
params_start.loc[("disutil_work", "yes_kids_p_educ_low"), cols] = [-0.70, False]
params_start.to_pickle("start_udpated.soepy.pkl")

# Setup 
adapter_kwargs["params"] = params_start
adapter_smm = SimulationBasedEstimationCls(**adapter_kwargs)
np.testing.assert_almost_equal(adapter_smm.fval, 275607.7585997434)

# Estimation
opt_kwargs["maxfun"] = 1

x0, bounds = prepare_optimizer_interface(params_start)
p_wrapper_numpy = partial(wrapper_numpy, params_start, adapter_smm)
rslt = pybob.solve(p_wrapper_numpy, x0, bounds=bounds, **opt_kwargs)
np.testing.assert_almost_equal(rslt.f, 275607.7585997434)