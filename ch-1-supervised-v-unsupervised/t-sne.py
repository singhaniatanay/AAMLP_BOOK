import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns

from sklearn import datasets
from sklearn import manifold

%matplotlib inline

data = datasets.fetch_openml('mnist_784',version=1,return_X_y=True)
pixel_vals,targets = data
targets = targets.astype(int)