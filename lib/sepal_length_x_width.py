import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

import load_data as ld

# Visualisation sepal_length x sepal_width
plt.plot(ld.setosa.sepal_length, ld.setosa.sepal_width, 'ro')
red_patch = mpatches.Patch(color='red', label='setosa')
plt.plot(ld.versicolor.sepal_length, ld.versicolor.sepal_width, 'go')
green_patch = mpatches.Patch(color='green', label='versicolor')
plt.plot(ld.virginica.sepal_length, ld.virginica.sepal_width, 'bo')
blue_patch = mpatches.Patch(color='blue', label='virginica')
plt.legend(handles=[red_patch, green_patch, blue_patch])
plt.xlabel('sepal length [cm]')
plt.ylabel('sepal width [cm]')
plt.show()
