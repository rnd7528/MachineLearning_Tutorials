#=========================================================
# begin Task 1 Template - SOLUTION
#
# Task 1: Use the following template to overlay linear, quadaratic, and cubic
# fits ontop of ground truth data.
# ground truth points are navy circles, groud truth solid line is light green
# linear, quadratic, cubic line colors are red, cyan, blue

import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import Ridge, LinearRegression
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
linear = LinearRegression()  #note this is not necessary
linear.fit(X_train, y_train)
predict_linear = linear.predict(X_test)

degree=1
poly1 = make_pipeline(PolynomialFeatures(degree), Ridge())
poly1.fit(X_train, y_train)
predict_poly1 = poly1.predict(X_test)
degree=2
poly2 = make_pipeline(PolynomialFeatures(degree), Ridge())
poly2.fit(X_train, y_train)
predict_poly2 = poly2.predict(X_test)
degree=3
poly3 = make_pipeline(PolynomialFeatures(degree), Ridge())
poly3.fit(X_train, y_train)
predict_poly3 = poly3.predict(X_test)

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

#plt.plot(x_test, predict_linear, color='black', linewidth=lw,
#             label="linear")
plt.plot(x_test, predict_poly1, color=colors[0], linewidth=lw,
             label="poly degree %d" % 1)
plt.plot(x_test, predict_poly2, color=colors[1], linewidth=lw,
             label="poly degree %d" % 2)
plt.plot(x_test, predict_poly3, color=colors[2], linewidth=lw,
             label="poly degree %d" % 3)
plt.legend(loc='upper left')
plt.show()
################################################################################
#                              END OF YOUR CODE                                #
################################################################################

# end of Task 1 Template - SOLUTION
#=========================================================