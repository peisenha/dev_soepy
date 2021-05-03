#!usr/bin/env python
import pickle as pkl

import pandas as pd
import numpy as np

from SimulationBasedEstimation import SimulationBasedEstimationCls
from moments import get_moments

df_start = pd.read_pickle("start.soepy.pkl")

adapter_kwargs = dict()
adapter_kwargs["weighting_matrix"] = pkl.load(open("weighting-matrix.pkl", "rb"))
adapter_kwargs["model_spec_init_file_name"] = "resources/model_spec_init.yml"
adapter_kwargs["moments_obs"] = pkl.load(open("observed-moments.pkl", "rb"))
adapter_kwargs["get_moments"] = get_moments
adapter_kwargs["params"] = df_start

adapter_smm = SimulationBasedEstimationCls(**adapter_kwargs)
np.testing.assert_almost_equal(adapter_smm.fval, 2366.4468131697445)
