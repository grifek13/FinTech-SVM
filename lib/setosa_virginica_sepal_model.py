import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import load_data as ld
from sklearn import svm

# input for the model
setosa_virginica = ld.training[ld.training.species != 'versicolor']
species_label = np.where(setosa_virginica['species'] == 'setosa', 0, 1) # to predict 0: setosa 1: virginica
setosa_virginica = np.asmatrix(setosa_virginica[['sepal_length', 'sepal_width']])

# fit the SVM model
model = svm.SVC(kernel='linear', C=1) # default value 1 (how much we want to penalize the misclassify of points)
model.fit(setosa_virginica, species_label)

# plot the hyperplane on the training set
w = model.coef_[0]
a = -w[0] / w[1]
xx = np.linspace(4, 8)
yy = a * xx - (model.intercept_[0]) / w[1]

# plot the training setosa and virginica
training_setosa = ld.training[ld.training.species == 'setosa']
training_virginica = ld.training[ld.training.species == 'virginica']

plt.plot(training_setosa.sepal_length, training_setosa.sepal_width, 'ro')
red_patch = mpatches.Patch(color='red', label='setosa')
plt.plot(training_virginica.sepal_length, training_virginica.sepal_width, 'bo')
blue_patch = mpatches.Patch(color='blue', label='virginica')
plt.plot(xx, yy)
plt.legend(handles=[red_patch, blue_patch])
plt.xlabel('sepal length [cm]')
plt.ylabel('sepal width [cm]')
print(plt.show())

# plot testset
test_setosa = ld.testing[ld.testing.species == 'setosa']
test_virginica = ld.testing[ld.testing.species == 'virginica']

plt.plot(training_setosa.sepal_length, training_setosa.sepal_width, 'ro')
plt.plot(training_virginica.sepal_length, training_virginica.sepal_width, 'bo')
plt.plot(test_setosa.sepal_length, test_setosa.sepal_width, 'rs')
plt.plot(test_virginica.sepal_length, test_virginica.sepal_width, 'bs')
red_patch = mpatches.Patch(color='red', label='setosa')
blue_patch = mpatches.Patch(color='blue', label='virginica')
plt.plot(xx, yy)
plt.legend(handles=[red_patch, blue_patch])
plt.xlabel('sepal length [cm]')
plt.ylabel('sepal width [cm]')
print(plt.show())
# To predict the species: model.predict([some sepal length, some sepal width])
# returns 0 when setosa, 1 when virginica.


