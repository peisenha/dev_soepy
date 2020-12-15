#!usr/bin/env python
import pickle
import time
import yaml
import sys
import os

import pandas as pd
import numpy as np

from SimulationBasedEstimation import SimulationBasedEstimationCls


base_dir = os.getcwd()
os.chdir("../../analysis/estimation/template-global-optimization")

sys.path.insert(0, ".")
from global_moments import get_moments

grid = np.linspace(1000, 80000, num=80, dtype=int)
rslt = pd.DataFrame(None, columns=["Fval", "Time"], index=grid)
rslt.index.name = "Agents"

base_spec = yaml.load(open("resources/model_spec_init.yml"), Loader=yaml.Loader)

for num_agents_sim in rslt.index.get_level_values("Agents"):
    base_spec["SIMULATION"]["num_agents_sim"] = num_agents_sim
    yaml.dump(base_spec, open('data.yml', "w"), default_flow_style=False)

    model_spec_fname = "data.yml"
    model_para_fname = "resources/model_params.pkl"

    weighting_matrix = pickle.load(open("resources/weighting_matrix_ones.pkl", "rb"))
    moments_obs = pickle.load(open("resources/moments_obs.pkl", "rb"))
    model_params = pd.read_pickle(model_para_fname)

    # We need to set up our criterion function.
    adapter_kwargs = dict()
    adapter_kwargs["model_spec_init_file_name"] = model_spec_fname
    adapter_kwargs["weighting_matrix"] = weighting_matrix
    adapter_kwargs["moments_obs"] = moments_obs
    adapter_kwargs["get_moments"] = get_moments
    adapter_kwargs["params"] = model_params

    adapter_smm = SimulationBasedEstimationCls(**adapter_kwargs)

    start = time.time()
    fval = adapter_smm.get_objective(model_params)
    finish = time.time()

    rslt.loc[num_agents_sim, "Time"] = finish - start
    rslt.loc[num_agents_sim, "Fval"] = fval

    rslt.to_pickle(f"{base_dir}/rslt-chatter.pkl")
