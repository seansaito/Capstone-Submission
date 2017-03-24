import tensorflow as tf
import numpy as np
import math
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import os
import datetime
from sklearn.cross_validation import train_test_split


def lrelu(x, leak=0.2, name="lrelu"):
    with tf.variable_scope(name):
        f1 = 0.5 * (1 + leak)
        f2 = 0.5 * (1 - leak)
        return f1 * x + f2 * abs(x)

# %%
def autoencoder(input_shape=[None, 784],
                n_filters=[1, 10, 10, 10],
                filter_sizes=[3, 3, 3],
                corruption=False):
    # %%
    # input to the network
    x = tf.placeholder(
        tf.float32, input_shape, name='x')


    # %%
    # ensure 2-d is converted to square tensor.
    if len(x.get_shape()) == 2:
        x_dim = np.sqrt(x.get_shape().as_list()[1])
        if x_dim != int(x_dim):
            raise ValueError('Unsupported input dimensions')
        x_dim = int(x_dim)
        x_tensor = tf.reshape(
            x, [-1, x_dim, x_dim, n_filters[0]])
    elif len(x.get_shape()) == 4:
        x_tensor = x
    else:
        raise ValueError('Unsupported input dimensions')
    current_input = x_tensor

    # %%
    # Optionally apply denoising autoencoder
    if corruption:
        current_input = corrupt(current_input)

    # %%
    # Build the encoder
    encoder_weights = []
    encoder_ops = []
    shapes = []
    for layer_i, n_output in enumerate(n_filters[1:]):
        n_input = current_input.get_shape().as_list()[3]
        shapes.append(current_input.get_shape().as_list())
        W = tf.Variable(
            tf.random_uniform([
                filter_sizes[layer_i],
                filter_sizes[layer_i],
                n_input, n_output],
                -1.0 / math.sqrt(n_input),
                1.0 / math.sqrt(n_input)))
        b = tf.Variable(tf.zeros([n_output]))
        encoder_weights.append(W)
        output = lrelu(
            tf.add(tf.nn.conv2d(
                current_input, W, strides=[1, 1, 1, 1], padding='SAME'), b))
        encoder_ops.append(output)
        current_input = output

    # %%
    # store the latent representation
    z = current_input
    encoder_weights.reverse()
    shapes.reverse()

    # %%
    # Build the decoder using the same weights
    for layer_i, shape in enumerate(shapes):
        W = encoder_weights[layer_i]
        b = tf.Variable(tf.zeros([W.get_shape().as_list()[2]]))
        output = lrelu(tf.add(
            tf.nn.conv2d_transpose(
                current_input, W,
                tf.pack([tf.shape(x)[0], shape[1], shape[2], shape[3]]),
                strides=[1, 1, 1, 1], padding='SAME'), b))
        current_input = output

    # %%
    # now have the reconstruction through the network
    y = current_input
    # cost function measures pixel-wise difference
    cost = tf.reduce_sum(tf.square(y - x_tensor))

    # %%
    return {'x': x, 'z': z, 'y': y, 'cost': cost, "encoder": encoder_ops}


# %%
def test_mnist(n_filters, filter_sizes):
    """Test the convolutional autoencder using MNIST."""
    # %%
    import tensorflow as tf
    import tensorflow.examples.tutorials.mnist.input_data as input_data
    import matplotlib.pyplot as plt

    # %%
    # load MNIST as before
    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
    mean_img = np.mean(mnist.train.images, axis=0)
    ae = autoencoder(n_filters=n_filters, filter_sizes=filter_sizes)

    # %%
    learning_rate = 0.01
    optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(ae['cost'])

    # %%
    # We create a session to use the graph
    config = tf.ConfigProto()
    config.gpu_options.allow_growth=True
    sess = tf.Session(config=config)
    sess.run(tf.initialize_all_variables())

    # %%
    # Fit all training data
    batch_size = 100
    n_epochs = 100
    step_size = 10
    for epoch_i in range(n_epochs):
        for batch_i in range(mnist.train.num_examples // batch_size):
            batch_xs, _ = mnist.train.next_batch(batch_size)
            train = np.array([img - mean_img for img in batch_xs])
            sess.run(optimizer, feed_dict={ae['x']: train})
        if epoch_i % step_size == 0:
            print(str(datetime.datetime.now()), epoch_i, sess.run(ae['cost'], feed_dict={ae['x']: train}))

    ae["session"] = sess
    
    return ae

import tensorflow.examples.tutorials.mnist.input_data as input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
mnist.train.next_batch(10)[0].shape
mean_img = np.mean(mnist.train.images, axis=0)
batch_xs, batch_ys = mnist.train.next_batch(10000)
train = np.array([img - mean_img for img in batch_xs])
y = [np.argmax(row) for row in batch_ys]
y_train = np.array(y)

# 3 layers

ae = test_mnist([1, 10, 10], [3, 3])

combined = []
sess = ae["session"]
batch_size = 100
for batch_i in range(train.shape[0] // batch_size):
    batch_xs = train[batch_i * batch_size:(batch_i + 1) * batch_size]
    layers = [sess.run(ae["encoder"][i], 
            feed_dict={ae['x']: batch_xs}) for i in range(len(ae["encoder"]))]
    ravels = (np.array([row.ravel() for row in layers[i]]) for i in range(len(ae["encoder"])))
    interm = np.hstack((ravels))
    combined.append(interm)
    
combined = np.vstack((combined))
print combined.shape

ks = [3, 5, 10, 20]

from sklearn.neighbors import KNeighborsClassifier

for k in ks:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(combined, y_train[:combined.shape[0]])
    print ("KNN Score with k=%i" % k,  knn.score(combined, y_train[:combined.shape[0]]))

