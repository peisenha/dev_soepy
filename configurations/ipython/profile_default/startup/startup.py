from IPython import get_ipython

ipython = get_ipython()

ipython.magic("load_ext autoreload")

ipython.magic("matplotlib inline")
ipython.magic("autoreload 2")