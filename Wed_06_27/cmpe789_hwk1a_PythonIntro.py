#=========================================================
#
# CMPE-789 Deep Learning
# Spring 2017
# Homework #1a  
#
# This file will orient you with some of the basic machine 
# learning capabilities of Python.  This code is Python 2.7
# Many of these examples in this file were inspired 
# from http://scikit-learn.org/stable/
#
# If you are not familiar with Python, first step through the file:
# http://cs231n.github.io/python-numpy-tutorial/#numpy-array-indexing
#
# Troubleshooting:
# Need to make sure libraries are up to date.  If using Anaconda:
#    conda update conda
#    conda update ipython ipython-notebook ipython-qtconsole
#
# Anaconda bug.  In Anaconda with Python 3.5.2, you may get error:
# "Intel MKL FATAL ERROR: Cannot load libmkl_avx.so or libmkl_def.so"
# Workaround:
#    conda install nomkl numpy scipy scikit-learn numexpr
#    See:  https://groups.google.com/forum/#!topic/gensim/PAqtRG6nL5w
#
#=========================================================

#=========================================================
#
# We start with an example on linear regression
# step through this section before continuing
#
#Linear Regression:
import numpy as np
from sklearn import linear_model, datasets
from sklearn.metrics import mean_squared_error
#from math import floor, ceil
import matplotlib.pyplot as plt
#Load Train and Test datasets
diabetes = datasets.load_diabetes()
X, y = diabetes.data, diabetes.target
y = y[:, np.newaxis] # create matrix versions of this array
X = X[:,np.newaxis,2] #extract third column from X

#Identify feature and response variable(s) and values must be numeric and numpy arrays
n_samples = len(diabetes.target)  #should be 442
x_train = X[:n_samples / 2]  #first half of samples (221 samples)
y_train = y[:n_samples / 2]  
x_test = X[n_samples / 2:]  #second half of samples (221 samples)
y_test = y[n_samples / 2:]  

# Create linear regression object
linear = linear_model.LinearRegression()
# Train the model using the training sets and check score
linear.fit(x_train, y_train)

#Equation coefficient and Intercept
print('Coefficient: \n', linear.coef_)
print('Intercept: \n', linear.intercept_)
#Predict Output
predicted= linear.predict(x_test)
#mean squared error
mse1 = sum((predicted-y_test)*(predicted-y_test))/len(y_test)
mse2 = np.mean((predicted-y_test) ** 2)
mse = mean_squared_error(y_test, predicted)  #or use built-in function
print("Mean squared error: %.2f" % mse )
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % linear.score(x_train, y_train))

#plot output
plt.plot(x_train, y_train,'ro')
plt.plot(x_test , linear.predict(x_test), color='blue',linewidth=3)
plt.xlabel('Input')
plt.ylabel('Output')
plt.title('Linear Fit')
plt.show()
#=========================================================


#=========================================================
# begin Task 1 Template
#
# Task 1: Use the following template to overlay linear, quadaratic, and cubic
# fits ontop of ground truth data.
# ground truth points are navy circles, groud truth solid line is light green
# linear, quadratic, cubic line colors are red, cyan, blue

import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline


def myCurve(x):
    """ function to approximate by polynomial interpolation"""
    return np.abs(x) + x * np.sin(x)

# generate training set
x_train = np.linspace(-4, 10, 50) #50 points from -4 to 10
y_train = myCurve(x_train)

# generate test points  
x_test = np.linspace(-3.3, 9.8, 50) #50 points from -3.3 to 9.8

# create matrix versions of these arrays
X_train = x_train[:, np.newaxis]
X_test = x_test[:, np.newaxis]

#plot ground truth
colors = ['red', 'cyan', 'blue']
lw = 2
plt.plot(x_test, myCurve(x_test), color='lightgreen', linewidth=lw,
         label="ground truth")
plt.scatter(x_train, y_train, color='green', s=30, marker='o', label="training points")


################################################################################
# TODO:                                                                        #
# insert code here to build linear, quadratic and cubic models                 #
# 1) create model with make_pipeline(PolynomialFeatures(degree), Ridge())      #
# 2) fit model with model.fit                                                  #
# 3) evaluate model with model.predict                                         #
#                                                                              #
################################################################################

################################################################################
#                              BEGIN YOUR CODE                                 #
################################################################################
pass
################################################################################
#                              END OF YOUR CODE                                #
################################################################################

################################################################################
# TODO:                                                                        #
# insert code below to plot linear, quadratic, and cubic results               #
# insert a legend in the upper left corner                                     #
#                                                                              #
################################################################################

################################################################################
#                              BEGIN YOUR CODE                                 #
################################################################################
pass
################################################################################
#                              END OF YOUR CODE                                #
################################################################################


# end of Task 1 Template
#=========================================================




#=========================================================
# We next have a classification example
# step through this section before continuing
# much of this inspried by:
# http://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_moons
from sklearn.svm import SVC

# create dataset
X, y = make_moons(noise=0.3, random_state=0)
X_train, X_test, y_train, y_test = \
        train_test_split(X, y, test_size=.4, random_state=42)

# mesh data for plots
h = .02  # step size in the mesh
x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

# plot the dataset 
plt.figure(1)
cm = plt.cm.RdBu
cm_bright = ListedColormap(['#FF0000', '#0000FF'])
plt.title("Input data")
# Plot the training points
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright)
# and testing points
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright, alpha=0.6)
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
#plt.xticks(())  #use to remove ticks
#plt.yticks(())


# build classifier
name = "Linear SVM"
classifier= SVC(kernel="linear", C=0.025)
classifier.fit(X_train, y_train)
score = classifier.score(X_test, y_test)

# plot mesh of classifier
plt.figure(2)
# Plot the decision boundary. For that, we will assign a color to each
# point in the mesh [x_min, x_max]x[y_min, y_max].
Z = classifier.decision_function(np.c_[xx.ravel(), yy.ravel()])

# Put the result into a color plot
Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z, cmap=cm, alpha=.8)

# Plot also the training points
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright)
# and testing points
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright,
           alpha=0.6)

plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
    
plt.title(name)
plt.text(xx.max() - .3, yy.min() + .3, ('%.2f' % score).lstrip('0'),
        size=15, horizontalalignment='right')
plt.show()
#=========================================================
 

#=========================================================
# begin Task 2 Template
#
# Task 2: Use the following template to create a 3x2 subplot using the 
# dataset from above.
# The plots, from first to last are: 
# input dataset; 
# KNN K=3; 
# linearSVM C=1; 
# RBF SVM C=0.025 gamma=0.1; 
# Random Forest max_depth=5;
# Neural network , 2 hidden layers each with 20 neurons, regularization L2 penalty=1 

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

names = ["Nearest Neighbors", "Linear SVM", "RBF SVM", "Random Forest", "NNet"]

################################################################################
# TODO:                                                                        #
#                                                                              #
# fill in the 5 classifiers below                                              #
#                                                                              #
################################################################################
################################################################################
#                              BEGIN YOUR CODE                                 #
################################################################################
classifiers = [
    ]
################################################################################
#                              END OF YOUR CODE                                #
################################################################################


X, y = make_moons(noise=0.3, random_state=0)
X_train, X_test, y_train, y_test = \
        train_test_split(X, y, test_size=.4, random_state=42)

h = .02  # step size in the mesh
x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

i = 1
plt.figure(1)
ax = plt.subplot((len(names)+1)/2,(len(names)+1)/((len(names)+1)/2),i) #3x2 subplot, start on 1st. Could also use ax = plt.subplot(311)

# just plot the dataset first
cm = plt.cm.RdBu
cm_bright = ListedColormap(['#FF0000', '#0000FF'])
ax.set_title("Input data")
# Plot the training points
ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright)
# and testing points
ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright, alpha=0.6)
ax.set_xlim(xx.min(), xx.max())
ax.set_ylim(yy.min(), yy.max())
#ax.set_xticks(())
#ax.set_yticks(())
i += 1

# iterate over classifiers
for name, classifier in zip(names, classifiers):
 
    ax = plt.subplot((len(names)+1)/2,(len(names)+1)/((len(names)+1)/2),i)
	
    ################################################################################
    # TODO:                                                                        #
    #                                                                              #
    # fill in code below to train the classifier, then assign a score              #
    #                                                                              #
    ################################################################################
	
	################################################################################
    #                              BEGIN YOUR CODE                                 #
    ################################################################################
    pass
    ################################################################################
    #                              END OF YOUR CODE                                #
    ################################################################################
 
     

    # Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x_min, x_max]x[y_min, y_max].
    if hasattr(classifier, "decision_function"):
        Z = classifier.decision_function(np.c_[xx.ravel(), yy.ravel()])
    else:
        Z = classifier.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    ax.contourf(xx, yy, Z, cmap=cm, alpha=.8)

    # Plot also the training points
    ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright)
    # and testing points
    ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright,
               alpha=0.6)

    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    #ax.set_xticks(())
    #ax.set_yticks(())
    
    ax.set_title(name)
    ax.text(xx.max() - .3, yy.min() + .3, ('%.2f' % score).lstrip('0'),
            size=15, horizontalalignment='right')
    i += 1

plt.tight_layout()

# end of Task 2 Template  
#=========================================================




#=========================================================
#
# Working with images
#
#
#The full example of handwritten digit is from: 
#http://scikit-learn.org/stable/auto_examples/classification/plot_digits_classification.html#sphx-glr-auto-examples-classification-plot-digits-classification-py

import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, svm, metrics
from sklearn.metrics import classification_report

#load the digits dataset
digits = datasets.load_digits()
print(digits.data)
digits.target
digits.images[0]  #each image is 8x8

# The data that we are interested in is made of 8x8 images of digits, let's
# have a look at the first 4 images, stored in the `images` attribute of the
# dataset.  If we were working from image files, we could load them using
# matplotlib.pyplot.imread.  Note that each image must have the same size. For these
# images, we know which digit they represent: it is given in the 'target' of
# the dataset.
images_and_labels = list(zip(digits.images, digits.target))
for index, (image, label) in enumerate(images_and_labels[:4]):
    plt.subplot(1, 4, index + 1)
    plt.axis('off')
    plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
    plt.title('Training: %i' % label)
	
# To apply a classifier on this data, we need to flatten the image, to
# turn the data into a (samples, feature) matrix:
n_samples = len(digits.images)  #should be 1797
#data = digits.images  #this would give 1797x8x8
data = digits.images.reshape((n_samples, -1))  #data will become 1797x64

# Create a classifier: a support vector classifier
classifier = svm.SVC(gamma=0.001, C=100)

# We learn the digits on the first half of the digits
classifier.fit(data[:n_samples / 2], digits.target[:n_samples / 2])

# Now predict the value of the digit on the second half, 899 of each:
expected = digits.target[n_samples / 2:]
predicted = classifier.predict(data[n_samples / 2:])

print("Classification report for classifier %s:\n%s\n"
      % (classifier, metrics.classification_report(expected, predicted)))
cm = metrics.confusion_matrix(expected, predicted)
print("Confusion matrix:\n%s" % cm)
print("Accuracy = %d%%\n" % (100*cm.diagonal().sum()/cm.sum()))  #should be 96%

images_and_predictions = list(zip(digits.images[n_samples / 2:], predicted))
for index, (image, prediction) in enumerate(images_and_predictions[:4]):
    plt.subplot(1, 4, index + 1)
    plt.axis('off')
    plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
    plt.title('Prediction: %i' % prediction)

plt.show()

#There are many hyperparameters you can tune to get better performance
#For example, the command classifier.get_param() returns:
#{'C': 100.0,
# 'cache_size': 200,
# 'class_weight': None,
# 'coef0': 0.0,
# 'degree': 3,
# 'gamma': 0.001,
# 'kernel': 'rbf',
# 'max_iter': -1,
# 'probability': False,
# 'random_state': None,
# 'shrinking': True,
# 'tol': 0.001,
# 'verbose': False}
#
# There are two built-in commands for doing hyperparameter optimization:
# GridSearchCV and RandomizedSearchCV

#
# GridSearchCV
# Example of GridSearchCV from http://scikit-learn.org/stable/auto_examples/model_selection/grid_search_digits.html#sphx-glr-auto-examples-model-selection-grid-search-digits-py
#
# Use param_grid to specify hyperparameter values
# Two search candidates stored in this example (linear and rbf)
#param_grid = [
#  {'C': [1, 10, 100, 1000], 'kernel': ['linear']},
#  {'C': [1, 10, 100, 1000], 'gamma': [0.001, 0.0001], 'kernel': ['rbf']},
# ]
#=========================================================

#=========================================================
# begin Task 3 Template  
#
# Task 3: Use the following template to implement GridSearchCV.
# For example, you can specify:
# tuned_parameters = [{'kernel': ['linear'], 'C': [0.1, 1, 10]},
#                    {'kernel': ['rbf'], 'C': [0.1, 1, 10]}]
# To explore SVM with linear and rbf kernels, evaluating the 
# parameter C over a range of values 

from sklearn import datasets, svm, metrics
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.svm import SVC

# Loading the Digits dataset
digits = datasets.load_digits()

# To apply an classifier on this data, we need to flatten the image, to
# turn the data in a (samples, feature) matrix:
n_samples = len(digits.images)
X = digits.images.reshape((n_samples, -1))
y = digits.target

# Split the dataset in two equal parts
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=<<fill in code here>>, random_state=0)

################################################################################
# TODO:                                                                        #
#                                                                              #
# Set the parameters for cross-validation                                      #
# fill in values so so that we evaluate two test cases:                        #
# 1) linear SVM with C= 0.1, 1, 10, 100                                        #
# 2) RBF SVM with C=0.1,1,10, 100, 1000 and gamma = 0.001, 0.0001              #
#                                                                              #
################################################################################

################################################################################
#                              BEGIN YOUR CODE                                 #
################################################################################
tuned_parameters = [{<<fill in code here>>},
                    {<<fill in code here>>}]
################################################################################
#                              END OF YOUR CODE                                #
################################################################################
	

score = 'accuracy' # could also be 'precision_macro', 'recall_micro', ...see http://scikit-learn.org/stable/modules/model_evaluation.html

print("# Tuning hyper-parameters for %s" % score)

clf = GridSearchCV(SVC(C=1), tuned_parameters, cv=5,
                   scoring='%s' % score)
clf.fit(X_train, y_train)

print("Best parameters set found on development set:")
print(clf.best_params_)
print("All scores on development set:")
means = clf.cv_results_['mean_test_score']
stds = clf.cv_results_['std_test_score']
for mean, std, params in zip(means, stds, clf.cv_results_['params']):
    print("%0.3f (+/-%0.03f) for %r"
          % (mean, std * 2, params))
  
print("Detailed classification report:")
print("The model is trained on the full development set.")
print("The scores are computed on the full evaluation set.")
y_true, y_pred = y_test, clf.predict(X_test)
print(classification_report(y_true, y_pred))
print("Confusion matrix:\n%s" % metrics.confusion_matrix(y_true, y_pred))
cm = metrics.confusion_matrix(y_true, y_pred)
print("Accuracy of best model = %d%% \n" % (100*cm.diagonal().sum()/cm.sum()))   

# end of Task 3 Template 
#========================================================= 




#=========================================================
# begin Task 4 Template   
#
# Task 4: Use the following template to contrast
# RandomizedSearchCV vs. GridSearchCV
# Example of RandomizedSearchCV from http://scikit-learn.org/stable/auto_examples/model_selection/randomized_search.html#sphx-glr-auto-examples-model-selection-randomized-search-py
#
# A computation budget is set via n_iter
# For each hyperparameter, either a distribution over possible values or 
# a list of discrete choices (which will be sampled uniformly) can be specified:
#{'C': scipy.stats.expon(scale=100), 'gamma': scipy.stats.expon(scale=.1),
#  'kernel': ['rbf'], 'class_weight':['balanced', None]}

import numpy as np

from time import time
from scipy.stats import randint as sp_randint

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.datasets import load_digits
from sklearn.ensemble import RandomForestClassifier

# get some data
digits = load_digits()
X, y = digits.data, digits.target

# build a classifier
clf = RandomForestClassifier(n_estimators=20)


# Utility function to report best scores
def report(results, n_top=3):
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                  results['mean_test_score'][candidate],
                  results['std_test_score'][candidate]))
            print("Parameters: {0}".format(results['params'][candidate]))
            print("")


# specify parameters and distributions to sample from
param_dist = {"max_depth": [3, None],
              "max_features": sp_randint(1, 11),
              "min_samples_split": sp_randint(2, 11),
              "min_samples_leaf": sp_randint(1, 11),
              "bootstrap": [True, False],
              "criterion": ["gini", "entropy"]}
# run randomized search
n_iter_search = 30
random_search = RandomizedSearchCV(clf, param_distributions=param_dist,
                                   n_iter=n_iter_search)
start = time()
random_search.fit(X, y)
print("RandomizedSearchCV took %.2f seconds for %d candidates"
      " parameter settings." % ((time() - start), n_iter_search))
report(random_search.cv_results_)


# Now compare above to GridSearchCV
# report top three models and time
# use below param_grid:
param_grid = {"max_depth": [3, None],
              "max_features": [1, 11],
              "min_samples_split": [2, 11],
              "min_samples_leaf": [1, 11],
              "bootstrap": [True, False],
              "criterion": ["gini", "entropy"]}

################################################################################
# TODO:                                                                        #
#                                                                              #
# Insert code here, compare performance from                                   #
# top three models from GridSearchCV along with                                #
# execution time                                                               #
#                                                                              #
################################################################################

################################################################################
#                              BEGIN YOUR CODE                                 #
################################################################################
pass
################################################################################
#                              END OF YOUR CODE                                #
################################################################################

# end of Task 4 Template 
#========================================================= 



 


