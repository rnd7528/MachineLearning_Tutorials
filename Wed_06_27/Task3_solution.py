
#=========================================================
# begin Task 3 Template  - SOLUTION
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
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0)

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
tuned_parameters = [{'kernel': ['linear'], 'C': [0.1, 1, 10, 100]},
                    {'kernel': ['rbf'], 'C': [0.1, 1, 10, 100, 1000],
                                        'gamma': [1e-3, 1e-4]}]
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

# end of Task 3 Template - SOLUTION
#=========================================================