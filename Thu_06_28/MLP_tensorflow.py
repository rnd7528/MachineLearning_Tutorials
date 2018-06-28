'''
When one learns how to program, there's a tradition that the first thing you do is print "Hello World."
Just like programming has Hello World, machine learning has MNIST.
MNIST is a simple computer vision dataset. It consists of images of handwritten digits. It also includes labels for each image, telling us which digit it is.

In this tutorial,
-Create a MLP neural network that is a model for recognizing MNIST digits, based on looking at every pixel in the image
-Use Tensorflow to train the model to recognize digits by having it "look" at thousands of examples
(and run our first Tensorflow session to do so)
-Check the model's accuracy with our test data

Reference: https://www.tensorflow.org/versions/r1.0/get_started/mnist/beginners
'''

import warnings
warnings.filterwarnings("ignore")
import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt

def main():
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

    print('Training Data stats:')
    print('Train Data Size: ' + str(mnist.train.images.shape))
    print('Train Data Size: ' + str(mnist.train.labels.shape))

    # Python optimisation variables
    learning_rate = 0.5
    epochs = 10
    batch_size = 100

    # declare the training data placeholders
    # input x - for 28 x 28 pixels = 784
    x = tf.placeholder(tf.float32, [None, 784])
    # now declare the output data placeholder - 10 digits
    y = tf.placeholder(tf.float32, [None, 10])

    ######## TASK 1 ###########
    # In this exercise, we will be builing MLP neural network in Tensorflow.
    # In this task we will build a 2 layer network of the following specifications
    # [Input] - [300 neurons] - [10 neurons (softmax output)]
    # the First layer is defined as shown below.


    # now declare the weights connecting the input to the hidden layer
    W1 = tf.Variable(tf.random_normal([784, 300], stddev=0.03), name='W1')
    b1 = tf.Variable(tf.random_normal([300]), name='b1')

    # calculate the output of the hidden1 layer
    hidden1 = tf.add(tf.matmul(x, W1), b1)
    hidden1_out = tf.nn.relu(hidden1)

######  Fill in the Code  ########

# Similar to the above varaibles, define varaibles for weights connecting hidden1 layer and out layer
# For task 2 these will become varailbes for weights weights connecting hidden1 layer and hidden2 layer
# Please fill Code HERE
    # W2 =
    # b2 =

##### Ignore next block for Task 1 #########
########### only do Task 2 after Task 1 is complete ###########
    ######## TASK 2 ###########
    # In this task we will expand the 2 layer network to 3 layer network of the following specifications
    # [Input] - [300 neurons] - [100 neurons] - [10 neurons (softmax output)]
    # the First layer is defined as shown below.

    ######  Fill in the Code  ########
    # Similarly calculate the output of hidden2 layer

    # hidden2 =
    # hidden2_out =

    # Similar to the above varaibles, define varaibles for weights connecting hidden2 layer and output layer
    # Please fill Code HERE
    # W3 =
    # b3 =

########### only do Task 2 after Task 1 is complete ###########
##### Ignore above block for Task 1 #########

    # now calculate the hidden layer output - in this case, let's use a softmax activated
    # output layer

    y_ = tf.nn.softmax(tf.add(tf.matmul(hidden1_out, W2), b2))

    # now let's define the cost function which we are going to train the model on
    y_clipped = tf.clip_by_value(y_, 1e-10, 0.9999999)
    cross_entropy = -tf.reduce_mean(tf.reduce_sum(y * tf.log(y_clipped)
                                                  + (1 - y) * tf.log(1 - y_clipped), axis=1))

    # add an optimiser
    optimiser = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cross_entropy)

    # finally setup the initialisation operator
    init_op = tf.global_variables_initializer()

    # define an accuracy assessment operation
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # add a summary to store the accuracy
    tf.summary.scalar('accuracy', accuracy)

    merged = tf.summary.merge_all()
    writer = tf.summary.FileWriter('./logs/mlp_tensorflow')
    # start the session
    with tf.Session() as sess:
        # initialise the variables
        sess.run(init_op)
        total_batch = int(len(mnist.train.labels) / batch_size)
        for epoch in range(epochs):
            avg_cost = 0
            for i in range(total_batch):
                batch_x, batch_y = mnist.train.next_batch(batch_size=batch_size)
                _, c = sess.run([optimiser, cross_entropy], feed_dict={x: batch_x, y: batch_y})
                avg_cost += c / total_batch
            print("Epoch:", (epoch + 1), "cost =", "{:.3f}".format(avg_cost))
            summary = sess.run(merged, feed_dict={x: mnist.test.images, y: mnist.test.labels})
            writer.add_summary(summary, epoch)

        print("\nTraining complete!")
        print("Now Testing on test images")

        print('Test Data stats:')
        print('Test Data Size: ' + str(mnist.test.images.shape))
        print('Test Data Size: ' + str(mnist.test.labels.shape))
        writer.add_graph(sess.graph)

        print('Test Accuracy:')
        writer.add_graph(sess.graph)
        print(sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels}))

        for i in range(10):
            plt.subplot(2, 5, i + 1)
            weight = sess.run(W1)[:, i]
            plt.title(i)
            plt.imshow(weight.reshape([28, 28]), cmap=plt.get_cmap('seismic'))
            frame1 = plt.gca()
            frame1.axes.get_xaxis().set_visible(False)
            frame1.axes.get_yaxis().set_visible(False)
        plt.savefig('./mnist_mlp_weights.png')
        #plt.show()

if __name__ == '__main__':
    main()