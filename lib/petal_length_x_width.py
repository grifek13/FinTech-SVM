import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

import load_data as ld

# Visualisation petal_length x petal_width
plt.plot(ld.setosa.petal_length, ld.setosa.petal_width, 'ro')
red_patch = mpatches.Patch(color='red', label='setosa')
plt.plot(ld.versicolor.petal_length, ld.versicolor.petal_width, 'go')
green_patch = mpatches.Patch(color='green', label='versicolor')
plt.plot(ld.virginica.petal_length, ld.virginica.petal_width, 'bo')
blue_patch = mpatches.Patch(color='blue', label='virginica')
plt.legend(handles=[red_patch, green_patch, blue_patch])
plt.xlabel('petal length [cm]')
plt.ylabel('petal width [cm]')
plt.show()
