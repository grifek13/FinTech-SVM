import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import load_data as ld

# Ratio of petal to sepal.
plt.plot(ld.setosa.petal_length / ld.setosa.petal_width, ld.setosa.sepal_length / ld.setosa.sepal_width, 'ro')
red_patch = mpatches.Patch(color='red', label='setosa')
plt.plot(ld.versicolor.petal_length / ld.versicolor.petal_width, ld.versicolor.sepal_length / ld.versicolor.sepal_width, 'go')
green_patch = mpatches.Patch(color='green', label='versicolor')
plt.plot(ld.virginica.petal_length / ld.virginica.petal_width, ld.virginica.sepal_length / ld.virginica.sepal_width, 'bo')
blue_patch = mpatches.Patch(color='blue', label='virginica')
plt.legend(handles=[red_patch, green_patch, blue_patch])
plt.xlabel('petal length / petal width')
plt.ylabel('sepal length / sepal width')
print(plt.show())
