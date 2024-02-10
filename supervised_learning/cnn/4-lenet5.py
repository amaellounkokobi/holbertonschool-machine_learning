#!/usr/bin/env python3
"""
This module contains :
Write a function that builds a modified version
of the LeNet-5 architecture using tensorflow

Function:
   def lenet5(x, y):
"""
import tensorflow.compat.v1 as tf


def lenet5(x, y):
    """
    Builds a modified version of the LeNet-5
    architecture using tensorflow

    Args:
       x is a tf.placeholder of shape (m, 28, 28, 1)
       containing the input images for the network

       m is the number of images

       y is a tf.placeholder of shape (m, 10)
       containing the one-hot labels for the network

    """
    # Init variables
    m, nx, ny, nc = x.shape
    classes = y.shape[1]

    w_init = tf.keras.initializers.VarianceScaling(scale=2.0)
    b_init = tf.compat.v1.zeros_initializer()

    ReLu = tf.nn.relu
    softmax = tf.nn.softmax

    # Create layers
    conv2D_1 = tf.layers.conv2d(x,
                                filters=6,
                                kernel_size=(5, 5),
                                padding='same',
                                activation=ReLu,
                                kernel_initializer=w_init,
                                bias_initializer=b_init)

    m_pool2D_1 = tf.layers.max_pooling2d(conv2D_1,
                                         (2, 2),
                                         (2, 2))

    conv2D_1 = tf.layers.conv2d(m_pool2D_1,
                                filters=16,
                                kernel_size=(5, 5),
                                padding='valid',
                                activation=ReLu,
                                kernel_initializer=w_init,
                                bias_initializer=b_init)

    m_pool2D_2 = tf.layers.max_pooling2d(conv2D_1,
                                         (2, 2),
                                         (2, 2))

    x_flatten = tf.layers.Flatten()(m_pool2D_2)

    l_Dense_1 = tf.layers.Dense(120,
                                activation=ReLu,
                                kernel_initializer=w_init,
                                bias_initializer=b_init)(x_flatten)

    l_Dense_2 = tf.layers.Dense(84,
                                activation=ReLu,
                                kernel_initializer=w_init,
                                bias_initializer=b_init)(l_Dense_1)

    y_pred = tf.layers.Dense(10,
                             activation=softmax,
                             kernel_initializer=w_init,
                             bias_initializer=b_init)(l_Dense_2)

    # Calculate loss
    loss = tf.losses.softmax_cross_entropy(y, y_pred)

    # Train AdamOptimizer
    train_op = tf.train.AdamOptimizer().minimize(loss)

    # Accuracy
    true_false = tf.equal(tf.argmax(y, axis=1),
                          tf.argmax(y_pred, axis=1))

    accuracy = tf.reduce_mean(tf.cast(true_false, tf.float32))

    return y_pred, train_op, loss, accuracy
