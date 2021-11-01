"""
Author: Jorge H. Cárdenas
University of Antioquia

"""
import os
import secrets


try:
    import jupyterthemes
except:
    os.system('pip install jupyterthemes')

    
    import jupyterthemes

try:
    import pylab
except:
    os.system('pip install pylab')
    
    import pylab
try:
    import tqdm
except:
    os.system('pip install tqdm')
    
    import tqdm
    
    
try:
    import arviz
except:
    os.system('pip install arviz')
    
    import arviz as az
    
try:
    import numpy as np
except:
    os.system('pip install numpy')
    
    import numpy as np

try:
    import corner
except:
    os.system('pip install corner')
    
    import corner

try:
    import scipy
except:
    os.system('pip install scipy')
    import scipy

try:
    import seaborn
except:
    os.system('pip install seaborn')
    import seaborn

try:
    from sklearn.neighbors import KernelDensity
except:
    os.system('pip install -U scikit-learn')
    from sklearn.neighbors import KernelDensity


try:
    from mpi4py import MPI

except:
    os.system('pip install mpi4py')
    from mpi4py import MPI

from tqdm import tqdm
import math

import matplotlib.pyplot as plt
import pandas as pd
from IPython.core.display import HTML
import seaborn as sns

from bokeh.plotting import figure, show
from scipy import stats
from scipy.stats import lognorm
from sklearn.utils import shuffle


from statsmodels.graphics.tsaplots import plot_acf  
