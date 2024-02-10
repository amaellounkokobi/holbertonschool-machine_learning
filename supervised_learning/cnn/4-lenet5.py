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
    # Define initializer for kernel weights
    initializer = tf.keras.initializers.VarianceScaling(scale=2.0)

    # Convolutional layer 1
    conv1 = tf.layers.conv2d(
        inputs=x,
        filters=6,
        kernel_size=(5, 5),
        padding='same',
        activation=tf.nn.relu,
        kernel_initializer=initializer,
        name='conv1'
    )

    # Max pooling layer 1
    pool1 = tf.layers.max_pooling2d(
        inputs=conv1,
        pool_size=(2, 2),
        strides=(2, 2),
        name='pool1'
    )

    # Convolutional layer 2
    conv2 = tf.layers.conv2d(
        inputs=pool1,
        filters=16,
        kernel_size=(5, 5),
        padding='valid',
        activation=tf.nn.relu,
        kernel_initializer=initializer,
        name='conv2'
    )

    # Max pooling layer 2
    pool2 = tf.layers.max_pooling2d(
        inputs=conv2,
        pool_size=(2, 2),
        strides=(2, 2),
        name='pool2'
    )

    # Flatten the output of the convolutional layers
    flattened = tf.layers.flatten(pool2)

    # Fully connected layer 1
    fc1 = tf.layers.dense(
        inputs=flattened,
        units=120,
        activation=tf.nn.relu,
        kernel_initializer=initializer,
        name='fc1'
    )

    # Fully connected layer 2
    fc2 = tf.layers.dense(
        inputs=fc1,
        units=84,
        activation=tf.nn.relu,
        kernel_initializer=initializer,
        name='fc2'
    )

    # Output layer
    logits = tf.layers.dense(
        inputs=fc2,
        units=10,
        kernel_initializer=initializer,
        name='output'
    )

    # Softmax activation
    softmax_output = tf.nn.softmax(logits)

    # Loss function
    loss = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=logits),
        name='loss'
    )

    # Accuracy
    correct_predictions = tf.equal(tf.argmax(softmax_output, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32), name='accuracy')

    # Training operation
    optimizer = tf.train.AdamOptimizer()
    train_op = optimizer.minimize(loss)

    return softmax_output, train_op, loss, accuracy
