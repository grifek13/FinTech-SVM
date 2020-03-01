# This script presents data-set of irises that we use while presenting SVM.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from random import (seed, shuffle)

# Let's load the dataset
iris = pd.read_csv('../data/iris.csv')
setosa = iris[iris.species == 'setosa']
versicolor = iris[iris.species == 'versicolor']
virginica = iris[iris.species == 'virginica']

# split data-set into training and testing set.
seed(0)                  # You can make sample using any seed you want.
N = len(iris.index)    # Number of records
T = N // 2               # size of training set
indexes = list(range(N))
shuffle(indexes)         # shuffle the indexes

training = iris.iloc[indexes[:T]]
testing = iris.iloc[indexes[T:]]
