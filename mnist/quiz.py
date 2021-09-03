"""
================================
Recognizing hand-written digits
================================

This example shows how scikit-learn can be used to recognize images of
hand-written digits, from 0-9.
"""

print(__doc__)

# Author: Gael Varoquaux <gael dot varoquaux at normalesup dot org>
# License: BSD 3 clause

# Standard scientific Python imports
import matplotlib.pyplot as plt

# Import datasets, classifiers and performance metrics
from sklearn import datasets, svm, metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

###############################################################################
# Digits dataset
# --------------
#
# The digits dataset consists of 8x8
# pixel images of digits. The ``images`` attribute of the dataset stores
# 8x8 arrays of grayscale values for each image. We will use these arrays to
# visualize the first 4 images. The ``target`` attribute of the dataset stores
# the digit each image represents and this is included in the title of the 4
# plots below.
#
# Note: if we were working from image files (e.g., 'png' files), we would load
# them using :func:`matplotlib.pyplot.imread`.

digits = datasets.load_digits()

_, axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 3))
for ax, image, label in zip(axes, digits.images, digits.target):
    ax.set_axis_off()
    ax.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
    ax.set_title('Training: %i' % label)

###############################################################################
# Classification
# --------------
#
# To apply a classifier on this data, we need to flatten the images, turning
# each 2-D array of grayscale values from shape ``(8, 8)`` into shape
# ``(64,)``. Subsequently, the entire dataset will be of shape
# ``(n_samples, n_features)``, where ``n_samples`` is the number of images and
# ``n_features`` is the total number of pixels in each image.
#
# We can then split the data into train and test subsets and fit a support
# vector classifier on the train samples. The fitted classifier can
# subsequently be used to predict the value of the digit for the samples
# in the test subset.

# flatten the images
#make new feature branch for this:feature/quiz1

#MAKE AT LEAST THREE DIFF VERSION(RESOLUTIONS)
#identify size of image then resize in atleast 3
n_samples = len(digits.images)
data = digits.images.reshape((n_samples, -1))

# Create a classifier: a support vector classifier
clf = svm.SVC(gamma=0.001)

#CHANGE RATIO
# Split data into 50% train and 50% test subsets
X_train1, X_test1, y_train1, y_test1 = train_test_split(
    data, digits.target, test_size=0.1, shuffle=False)
X_train2, X_test2, y_train2, y_test2 = train_test_split(
    data, digits.target, test_size=0.3, shuffle=False)
X_train3, X_test3, y_train3, y_test3 = train_test_split(
    data, digits.target, test_size=0.5, shuffle=False)
#OUTPUT TABLE: DIFF IMAGE SIZE WITH DIFF IMG SIZE: ACCURACY

# 9 entries
#eg 64*64 | 90:10 | 76%


# Learn the digits on the train subset
clf.fit(X_train1, y_train1)
predicted1 = clf.predict(X_test1)
# Learn the digits on the train subset
clf.fit(X_train2, y_train2)
predicted2 = clf.predict(X_test2)
# Learn the digits on the train subset
clf.fit(X_train3, y_train3)
predicted3 = clf.predict(X_test3)

###############################################################################
# Below we visualize the first 4 test samples and show their predicted
# digit value in the title.

_, axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 3))
for ax, image, prediction in zip(axes, X_test1, predicted1):
    ax.set_axis_off()
    image = image.reshape(8, 8)
    ax.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
    ax.set_title(f'Prediction: {prediction}')

_, axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 3))
for ax, image, prediction in zip(axes, X_test2, predicted2):
    ax.set_axis_off()
    image = image.reshape(8, 8)
    ax.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
    ax.set_title(f'Prediction: {prediction}')

_, axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 3))
for ax, image, prediction in zip(axes, X_test3, predicted3):
    ax.set_axis_off()
    image = image.reshape(8, 8)
    ax.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
    ax.set_title(f'Prediction: {prediction}')

###############################################################################
# :func:`~sklearn.metrics.classification_report` builds a text report showing
# the main classification metrics.

#print(f"Classification report for classifier {clf}:\n"
      #f"{metrics.classification_report(y_test, predicted)}\n")

acc1 = accuracy_score(y_test1, predicted1)
acc2=accuracy_score(y_test2, predicted2)
acc3=accuracy_score(y_test3, predicted3)
###############################################################################
# We can also plot a :ref:`confusion matrix <confusion_matrix>` of the
# true digit values and the predicted digit values.

#disp = metrics.plot_confusion_matrix(clf, X_test, y_test)
#disp.figure_.suptitle("Confusion Matrix")
#print(f"Confusion matrix:\n{disp.confusion_matrix}")

#plt.show()
print('Ratio     Accuracy')
print('0.1',"     ",acc1)
print('0.3',"     ",acc1)
print('0.5',"     ",acc1)
