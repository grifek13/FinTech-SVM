import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import load_data as ld
from sklearn import svm

# input for the model (more than 2 classes)
input_data = np.asmatrix(ld.training[['petal_length', 'petal_width']])
input_species = np.array(list(map(lambda x: {'setosa': 0, 'versicolor': 1, 'virginica': 2}.get(x), ld.training['species'])))

# fit the SVM model
KERNEL = 'linear' # possible kernels: linear, rbf, poly,
model = svm.SVC(kernel=KERNEL, decision_function_shape='ovo') # (ovr) one versus rest | (ovo) one versus one
model.fit(input_data, input_species)

# plot the hyper-planes on the training set
x_min, x_max = input_data[:, 0].min() - 1, input_data[:, 0].max() + 1
y_min, y_max = input_data[:, 1].min() - 1, input_data[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))

res = model.predict(np.c_[xx.ravel(), yy.ravel()])
res = res.reshape(xx.shape)
plt.contour(xx, yy, res)

# plot training
training_setosa = ld.training[ld.training.species == 'setosa']
training_versicolor = ld.training[ld.training.species == 'versicolor']
training_virginica = ld.training[ld.training.species == 'virginica']
plt.plot(training_setosa.petal_length, training_setosa.petal_width, 'ro')
red_patch = mpatches.Patch(color='red', label='setosa')
plt.plot(training_versicolor.petal_length, training_versicolor.petal_width, 'go')
green_patch = mpatches.Patch(color='green', label='versicolor')
plt.plot(training_virginica.petal_length, training_virginica.petal_width, 'bo')
blue_patch = mpatches.Patch(color='blue', label='virginica')
plt.legend(handles=[red_patch, green_patch, blue_patch])
plt.xlabel('petal length [cm]')
plt.ylabel('petal width [cm]')
print(plt.show())

# plot tests
plt.contour(xx, yy, res)
test_setosa = ld.testing[ld.testing.species == 'setosa']
test_versicolor = ld.testing[ld.testing.species == 'versicolor']
test_virginica = ld.testing[ld.testing.species == 'virginica']
plt.plot(training_setosa.petal_length, training_setosa.petal_width, 'ro')
plt.plot(test_setosa.petal_length, test_setosa.petal_width, 'rs')
red_patch = mpatches.Patch(color='red', label='setosa')
plt.plot(training_versicolor.petal_length, training_versicolor.petal_width, 'go')
plt.plot(test_versicolor.petal_length, test_versicolor.petal_width, 'gs')
green_patch = mpatches.Patch(color='green', label='versicolor')
plt.plot(training_virginica.petal_length, training_virginica.petal_width, 'bo')
plt.plot(test_virginica.petal_length, test_virginica.petal_width, 'bs')
blue_patch = mpatches.Patch(color='blue', label='virginica')
plt.legend(handles=[red_patch, green_patch, blue_patch])
plt.xlabel('petal length [cm]')
plt.ylabel('petal width [cm]')
print(plt.show())
