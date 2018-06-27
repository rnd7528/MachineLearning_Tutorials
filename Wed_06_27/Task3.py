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

