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

    w_init = tf.keras.initializers.VarianceScaling(scale=2.0)

    # Create layers
    conv2D_1 = tf.layers.conv2d(inputs=x,
                                filters=6,
                                kernel_size=(5, 5),
                                padding='same',
                                activation=tf.nn.relu,
                                kernel_initializer=w_init)

    m_pool2D_1 = tf.layers.max_pooling2d(conv2D_1,
                                         pool_size=(2, 2),
                                         strides=(2, 2))

    conv2D_1 = tf.layers.conv2d(inputs=m_pool2D_1,
                                filters=16,
                                kernel_size=(5, 5),
                                padding='valid',
                                activation=tf.nn.relu,
                                kernel_initializer=w_init)

    m_pool2D_2 = tf.layers.max_pooling2d(inputs=conv2D_1,
                                         pool_size=(2, 2),
                                         strides=(2, 2))

    x_flatten = tf.layers.flatten(m_pool2D_2)

    l_Dense_1 = tf.layers.Dense(units=120,
                                activation=tf.nn.relu,
                                kernel_initializer=w_init)(x_flatten)

    l_Dense_2 = tf.layers.Dense(units=84,
                                activation=tf.nn.relu,
                                kernel_initializer=w_init)(l_Dense_1)

    y_pred = tf.layers.Dense(units=10,
                             kernel_initializer=w_init)(l_Dense_2)

    output_softmax = tf.nn.softmax(y_pred)

    # Calculate loss
    loss = tf.losses.softmax_cross_entropy(onehot_labels=y, logits=y_pred)

    # Train AdamOptimizer
    train_op = tf.train.AdamOptimizer().minimize(loss)

    # Accuracy
    true_false = tf.equal(tf.argmax(y, axis=1),
                          tf.argmax(output_softmax, axis=1))

    accuracy = tf.reduce_mean(tf.cast(true_false, tf.float32))

    return output_softmax, train_op, loss, accuracy
