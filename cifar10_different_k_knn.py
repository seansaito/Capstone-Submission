import tensorflow as tf
import numpy as np
import math
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import os
from sklearn.cross_validation import train_test_split
from tensorflow.models.image.cifar10 import cifar10_input
from keras.datasets import cifar10
import datetime

def autoencoder(input_shape=[None, 32, 32, 3],
                n_filters=[3, 3, 3, 3],
                filter_sizes=[5, 5, 5],
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
        output = tf.nn.sigmoid(
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
        output = tf.nn.sigmoid(
            tf.add(tf.nn.conv2d_transpose(
                current_input, W,
                tf.pack([tf.shape(x)[0], shape[1], shape[2], shape[3]]),
                strides=[1, 1, 1, 1], padding='SAME'), b))
        current_input = output
        
    decoder = current_input

    # %%
    # now have the reconstruction through the network
    y = current_input
    # cost function measures pixel-wise difference
    cost = tf.reduce_sum(tf.square(y - x_tensor))

    # %%
    return {'x': x, 'z': z, 'y': y, 'cost': cost, 
            "encoder": encoder_ops, "decoder": decoder}


# %%
def test_cifar(n_filters, filter_sizes):
    """Test the convolutional autoencder using MNIST."""
    # %%
    import tensorflow as tf
    from keras.datasets import cifar10
    import matplotlib.pyplot as plt

    # %%
    # load CIFAR10
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()
    X_train = X_train / 255.
    X_test = X_test / 255.
    ae = autoencoder(n_filters=n_filters, filter_sizes=filter_sizes)

    # %%
    learning_rate = 0.001
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
        for batch_i in range(X_train.shape[0] // batch_size):
            batch_xs = X_train[batch_i * batch_size:(batch_i + 1) * batch_size]
            train = batch_xs
            sess.run(optimizer, feed_dict={ae['x']: train})
        if epoch_i % step_size == 0:
            print(str(datetime.datetime.now()), epoch_i, sess.run(ae['cost'], feed_dict={ae['x']: train}))

    ae["session"] = sess
    
    return ae

ae = test_cifar(n_filters=[3,3,3,3], filter_sizes=[5,5,5])

num_examples = 10000

(X_train, y_train), (X_test, y_test) = cifar10.load_data()
X_train = X_train / 255.
X_test = X_test / 255.
train = X_train[:num_examples]
y_train = y_train.ravel()
y_test = y_test.ravel()

sess = ae["session"]
layers = [sess.run(ae["encoder"][i], 
        feed_dict={ae['x']: train}) for i in range(len(ae["encoder"]))]

ravels = (np.array([row.ravel() for row in layers[i]]) for i in range(len(ae["encoder"])))
combined = np.hstack(ravels)

y = list(y_train)[:10000]

ks = [3, 5, 10, 20]

from sklearn.neighbors import KNeighborsClassifier

for k in ks:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(combined, y)
    print ("KNN Score with k=%i" % k,  knn.score(combined, y))
