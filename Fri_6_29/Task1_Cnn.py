import warnings
warnings.filterwarnings("ignore")
import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import math
import pdb



def create_new_conv_layer(input_data, num_input_channels, num_filters, filter_shape, pool_shape, name):
    # setup the filter input shape for tf.nn.conv_2d
    conv_filt_shape = [filter_shape[0], filter_shape[1], num_input_channels, num_filters]

    # initialise weights and bias for the filter
    weights = tf.Variable(tf.truncated_normal(conv_filt_shape, stddev=0.03), name=name+'_W')
    bias = tf.Variable(tf.truncated_normal([num_filters]), name=name+'_b')

    # setup the convolutional layer operation
    # using the tensorflow function, tf.nn.conv2d
    out_layer = tf.nn.conv2d(input_data, weights, [1, 1, 1, 1], padding='SAME')

    # add the bias
    out_layer += bias

    # apply a ReLU non-linear activation
    out_layer = tf.nn.relu(out_layer)

    # now perform max pooling
    # ksize is the argument which defines the size of the max pooling window (i.e. the area over which the maximum is
    # calculated).  It must be 4D to match the convolution - in this case, for each image we want to use a 2 x 2 area
    # applied to each channel
    ksize = [1, pool_shape[0], pool_shape[1], 1]
    # strides defines how the max pooling area moves through the image - a stride of 2 in the x direction will lead to
    # max pooling areas starting at x=0, x=2, x=4 etc. through your image.  If the stride is 1, we will get max pooling
    # overlapping previous max pooling areas (and no reduction in the number of parameters).  In this case, we want
    # to do strides of 2 in the x and y directions.
    strides = [1, 2, 2, 1]

    # then pool using tensorflow tf.nn.max_pool function and the required parameters.
    out_layer = tf.nn.max_pool(out_layer, ksize=ksize, strides=strides, padding='SAME')

    return out_layer


def run_cnn():
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    
    print('Training Data stats:')
    print('Train Data Size: ' + str(mnist.train.images.shape))
    print('Train Data Size: ' + str(mnist.train.labels.shape))
    
    # Python optimization variables
    learning_rate = 0.0001
    epochs = 10
    batch_size = 50

    # declare the training data placeholders
    # input x - for 28 x 28 pixels = 784 - this is the flattened image data that is drawn from mnist.train.nextbatch()
    x = tf.placeholder(tf.float32, [None, 784])
    # reshape the input data so that it is a 4D tensor.  The first value (-1) tells function to dynamically shape that
    # dimension based on the amount of data passed to it.  The two middle dimensions are set to the image size (i.e. 28
    # x 28).  The final dimension is 1 as there is only a single colour channel i.e. grayscale.  If this was RGB, this
    # dimension would be 3
    x_shaped = tf.reshape(x, [-1, 28, 28, 1])
    # now declare the output data placeholder - 10 digits
    y = tf.placeholder(tf.float32, [None, 10])

###### Task 1 ########
###### Fill in the Code #####
    # create some convolutional layers
    # using the function 'create_new_conv_layer' which is defined above.
    # Pass the required parameters to the function
    # input_to_the_layer, num_input_channels, num_filters, filter_shape, pool_shape, name
    # filter_shape is the size of the conv. filter.
    # pool_shape is the size of pool filter.
    
    # layer1 = create_new_conv_layer(input variable , no. of input channels, number of output filters, Conv. filter size array (eg: [3,3] or [5,5], etc), pool filter size array (eg: [2,2]), name for the layer)
    # layer2 = 
###############################

    # flatten the output ready for the fully connected output stage - after two layers of stride 2 pooling, we go
    # from 28 x 28, to 14 x 14 to 7 x 7 x,y co-ordinates, but with 64 output channels.  To create the fully connected,
    # "dense" layer, the new shape needs to be [-1, 7 x 7 x 64]
    flattened = tf.reshape(layer2, [-1, 7 * 7 * 64])

###### Fill in the Code #######
    # setup some weights and bias values for this dense layer, then activate with ReLU, similar to the weights created below for second layer.
    # We now build a dense layer of size [flattened] neurons - 1000 neurons - 10 output neurons 

    # wd1 =
    # bd1 =
    # dense_layer1 =
    # dense_layer1 =    
##################################################

#Make sure you pass the previous dense layer to the dense_layer2
    # another layer with softmax activations
    wd2 = tf.Variable(tf.truncated_normal([1000, 10], stddev=0.03), name='wd2')
    bd2 = tf.Variable(tf.truncated_normal([10], stddev=0.01), name='bd2')
    dense_layer2 = tf.matmul(dense_layer1, wd2) + bd2
    y_ = tf.nn.softmax(dense_layer2)

    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=dense_layer2, labels=y))

    # add an optimizer
    optimiser = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cross_entropy)

    # define an accuracy assessment operation
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # setup the initialisation operator
    init_op = tf.global_variables_initializer()

    # setup recording variables
    # add a summary to store the accuracy
    tf.summary.scalar('accuracy', accuracy)
    tf.summary.scalar('loss', cross_entropy)

    merged = tf.summary.merge_all()
    writer = tf.summary.FileWriter('./logs/cnn_tensorflow')
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
            test_acc = sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels})
            print("Epoch:", (epoch + 1), "cost =", "{:.3f}".format(avg_cost), " test accuracy: {:.3f}".format(test_acc))
            summary = sess.run(merged, feed_dict={x: mnist.test.images, y: mnist.test.labels})
            writer.add_summary(summary, epoch)

        print("\nTraining complete!")
        
        print("Now Testing on test images")
        print('Test Data stats:')
        print('Test Data Size: ' + str(mnist.test.images.shape))
        print('Test Data Size: ' + str(mnist.test.labels.shape))
        
        print('\nTest Accuracy:')
        writer.add_graph(sess.graph)
        print(sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels}))
 
################ Task 2 ################

# a. Change the layer number and see their activations maps.
# b. Change the selected test image number for visualization of different class.
 
        # Change image number in below line
        imageToUse = mnist.test.images[0]
        # Change layer number in below line
        units = sess.run(layer1,feed_dict={x:np.reshape(imageToUse,[1,784],order='F')})
        filters = units.shape[3]
        plt.figure(1, figsize=(20,20))
        n_columns = 6
        n_rows = math.ceil(filters / n_columns) + 1
        ImName = './mnist_cnn_weights.png'
        for i in range(filters):
            plt.subplot(n_rows, n_columns, i+1)
            plt.title('Filter ' + str(i))
            plt.imshow(units[0,:,:,i], interpolation="nearest", cmap="gray")
            plt.savefig(ImName)

            
            
 ############# Task 3 ######################
 
 # Running Tensorboard for visualization of training.
 # In Command Window, enter the following command:
 # tensorboard --logdir='./logs/cnn_tensorflow' --port=8080
 # Then you shall see a URL on your screen. Copy paste that URL in the browser on local machine.
 
 ###########################################


if __name__ == "__main__":
    run_cnn()
