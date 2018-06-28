
# coding: utf-8

#  Tensorflow is arguably the most popular deep learning framework out there. It is designed to serve a diverse population which includes researchers, developers, enthusiasts, artists etc.
# 
#  In this exercise, we are going to learn basic tensorflow design concepts.

# 1. Verify TensorFlow installation
# 
# Run the following commands in a Python terminal:

import tensorflow as tf
import numpy as np
hello = tf.constant('Hello, TensorFlow!')
sess = tf.InteractiveSession()
print(sess.run(hello))


# # 2.	TensorFlow operations
# 
# TensorFlow:
#   - Represents computations as graphs.
#   - Executes graphs in the context of Sessions.
#   - Represents data as tensors.
#   - Maintains state with Variables.
#   - Uses feeds and fetches to get data into and out of arbitrary operations.
# 
# Unlike other deep learning frameworks you have used so far, 
# TensorFlow does not use an imperative style of programming. 
# Hence, it can be difficult to prototype your model if you are 
# new to TensorFlow. Also note that, TensorFlow development is
# very active in GitHub and is constantly changing. Hence, it 
# would be beneficial for you if you check the Issues and 
# Release notes in TensorFlow GitHub from time to time. 


# # Example:
# 
# 
# import tensorflow as tf		         			# import tensorflow 
# sess = tf.InteractiveSession()	         		# define a session
# node1 = tf.constant([[3.0, 2.0]], tf.float32)	    # define a constant
# print (node1.eval())	                            # evaluate the tensor and print its value
# print (node1.get_shape())	                	    # print shape
# print (node1.dtype)			                    # print type

 
# 2.1	Use the get_shape(), dtype and eval() operations to get the shape, type and value given tensors.

# a) A= tf.constant([3.0])

# b) A= tf.constant([3])

# c) A= tf.constant(1.0, shape=[3, 4])

# d) A= tf.constant(np.reshape(np.arange(2.0, 8.0, dtype=np.float32), (2, 3)))


# 2.2	Convert the following numpy expressions into TensorFlow operations
#       and use the eval() function to get the value. Hint: Refer to the 
#       Tensorflow Python API. You’ll find similarly named functions.


# a) A=  np.linspace(2.0, 3.0, num=5)

# b) A= np.stack((np.array([1, 2, 3]), np.array([2, 3, 4])))

# c) A= np.reshape(np.ones((2,2)), (1,4))

# d) A= np.dot(np.ones((2,2)), np.zeros((2,2)))


# 2.3 Control Flow Operations

# a). Execute the following code :

sess = tf.InteractiveSession()
i = tf.constant(0)
c = lambda i: tf.less(i, 10)
b = lambda i: tf.add(i, 1)
r = tf.while_loop(c, b, [i]) #repeat body (b), #while condition (c) id true
r.eval()

# b) Write the following while loop in TensorFlow:
# 
# Basic Python Code: 
#  i = 10
#  while (i > 0):
#      i--
#  print (i)


# c). Execute the following code : 

sess = tf.InteractiveSession()
x = tf.constant(2)
y = tf.constant(5)
def f1():
    return tf.multiply(x, 17)
def f2():
    return tf.add(y, 23)
# if x < y, execute f1, else execute f2
r = tf.cond(tf.less(x, y), f1, f2)
r.eval()


#  d) Write the following if else statement in TensorFlow:

# Basic Python Code: 

#  x = 5.0
#  y = 2.0
#  if (x > y):
#      print (x - 2.0)
#  else:
#      print (y / 2.0)
