#!usr/bin/env python
from functools import partial
import pickle as pkl

import pybobyqa as pybob
import pandas as pd
import numpy as np

from SimulationBasedEstimation import SimulationBasedEstimationCls

from pybobyqa_auxiliary import prepare_optimizer_interface
from pybobyqa_auxiliary import wrapper_numpy
from moments import get_moments

df_start = pd.read_pickle("start.soepy.pkl")

# We need to set up our criterion function.
adapter_kwargs = dict()
adapter_kwargs["weighting_matrix"] = pkl.load(open("weighting_matrix.pkl", "rb"))
adapter_kwargs["model_spec_init_file_name"] = "resources/model_spec_init.yml"
adapter_kwargs["moments_obs"] = pkl.load(open("observed_moments.pkl", "rb"))
adapter_kwargs["get_moments"] = get_moments
adapter_kwargs["params"] = df_start

adapter_smm = SimulationBasedEstimationCls(**adapter_kwargs)
np.testing.assert_almost_equal(adapter_smm.fval, 16375.422219820168)
