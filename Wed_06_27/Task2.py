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