import soepy

from soepy.test.random_init import random_init
from soepy.python.pre_processing.model_processing import transform_old_init_dict_to_df
from soepy.python.pre_processing.model_processing import read_model_params_init
random_init()

df = transform_old_init_dict_to_df("test.soepy.yml")
rslt = read_model_params_init(df)
