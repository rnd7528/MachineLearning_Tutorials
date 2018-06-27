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
