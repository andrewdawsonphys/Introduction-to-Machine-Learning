#!/usr/bin/env python
# coding: utf-8

#GOAL: Build a machine learning model that can learn from measurements of the irises whose species is unknown,
#so that we can predict the species for a new iris

# As we want to predict one of several options (the species of iris) This is an example of a classification problem
# The possible outputs (different species of irises) are called classes.
# Each iris in the dataset belongs to one of three classes, so this problem is a three-class classification problem

# Desired output for a single data point (an iris) is the species of the flower. For a particular data point,
# the species it belongs to is called its label

from sklearn.datasets import load_iris
iris_dataset = load_iris()

print("Keys of iris_dataset: \n{}".format(iris_dataset.keys()))


#DESCR is a short description of the data set
print(iris_dataset['DESCR'][:193] + "\n....")

#says it has 50 data points in each class (3 species of iris) and each data point it has 4 labels


# target_names is an array of strings, containing the species of flower that we want to predict

print("Target names: {}".format(iris_dataset['target_names']))


# The value of feature_names is a list of strings, giving the description of each feature
print("Feature names: \n{}".format(iris_dataset['feature_names']))


print("Type of data: {}".format(type(iris_dataset['data'])))


# The rows in the data array correspond to flowers, while the columns represent the four measurements 
#that were taken for each flower
print('Shape of data: {}'.format(iris_dataset['data'].shape))

#Each column (flower) is a sample in machine learning, and their properties are called features

# Will print the first 5 samples in the data set
print("First five columns of data \n{}".format(iris_dataset['data'][:5]))

# can see all flowers have a petal width of 0.2cm but the first flower has the longest stepal of 5.1cm


# The target array contains the species of each of the flowers that were measured, also as a NumPy array:

print("Type of target: {}".format(type(iris_dataset['target'])))

#target is a one dimensional array with one entry per flower

print("Shape of target: {}".format(iris_dataset['target'].shape))

# The species are encoded as integers from 0 to 2
print("Target:\n{}".format(iris_dataset['target']))

# 0 means Setosa
# 1 means versicolor
# 2 means virginica

''' To assess the models perfomance, we show it new data (data which it hasn't seen before) for which we have labels.
This is usually done by splitting the labelled data into two parts.

One part of the data is used as a training set and the other part is our test data. A good rule of thumb is to split
the data 75% training data and 25% test data(which also inclued the remaining labels)'''

# Calling train_test_split on our data and assign the outputs using this nomenclature
# data is normally denoted with X, while labels are denoted as y
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(iris_dataset['data'], iris_dataset['target'], random_state=0)

# train_test_split shuffles the dataset using a pseudorandom number generator as if we'd just took the last 25% we would have
# just gotten the label 2 out.

# the output of train_test_split is the four arrays specified before. X_train contains 75% of the rows of the dataset,
#and X_test contains the remaining 25%.


print("X_train shape : {}".format(X_train.shape))
print("y_train shape : {}".format(y_train.shape))


print("X_test shape : {}".format(X_test.shape))
print("y_test shape : {}".format(y_test.shape))

from pandas.plotting import scatter_matrix
import mglearn
# create dataframe from data in X_train
# label the column using the strings in iris_dataset.features_names
grr = scatter_matrix(iris_dataframe, c=y_train, figsize=(15,15), marker='o', hist_kwds={'bins': 20}, s=60, alpha=9, cmap=mglearn.cm3)

# See from the plot, the three classes seem to be relatively well seperated using the sepal and petal measurements.
# This means that a machine learning model will likely be able to learn to seperate them.

# Building the Model

# Here we will use a k-nearest neighbour classifier
# Building this model only consists of storing the training set.

#To make predictions for a new data point, the algorithm finds the point in the training set that is closest to the new point.
# Then it assigns the label of this training point to the new data point


from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=1)
# knn how encapsulates the algorithm that will be used to build the model. KNeighborsClassifier is the object

# To build the model on the training set, we call the fit method of the knn oject, which takes an argument of NumPy array
# X_train containing the training data and the NumPy array y_train of the corresponding training labels:

knn.fit(X_train,y_train);


# We can now make some new predictions using this model on new data for which we might not know the correct lebels.
# Imagine we found an iris in the wild with a sepal length of 5cm, a sepal width of 2.9cm a petal length of 1cm, 
# and a petal width of 0.2cm. What species of iris could this be?

import numpy as np

X_new = np.array([[5,2.9,1,0.2]])
print("X_new.shape: {}".format(X_new.shape))

# Single flower into a row in a two dimentioanl NumPy array


# To call the prediction, we call the predict method of the knn object:

prediction = knn.predict(X_new)
print("Prediction: {}".format(prediction))
print("Predicted target name: {}".format(iris_dataset['target_names'][prediction]))

# Our model predicts that this new iris belongs to the class 0. How can we trust our model?


# EVALUATING OUR MODEL

# This is where the test set comes into play. We can measure how well the model works by computing the accuracy, which is 
# the fraction of flowers for which the right species was predicted:

y_pred = knn.predict(X_test)
print("Test set predictions:\n{}".format(y_pred))

print("Test set score: {:.2f}".format(np.mean(y_pred== y_test)))


# or we can use the score method of the knn object, which will compute the test set accuracy for us:
print("Test set score: {:.2f}".format(knn.score(X_test,y_test)))

