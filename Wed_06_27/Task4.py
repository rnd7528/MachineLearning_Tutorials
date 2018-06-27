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