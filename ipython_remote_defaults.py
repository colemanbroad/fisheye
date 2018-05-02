from ipython_defaults import *

## This is safe to use on machine without $DISPLAY env var (headless furiosa)
import matplotlib
matplotlib.use('Agg') # this is already done in remote bashrc!
import matplotlib.pyplot as plt
