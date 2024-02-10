#!/usr/bin/env python3
"""
This module contains :
Write a function that builds a modified version
of the LeNet-5 architecture using keras

Function:
   def lenet5(X):
"""
import tensorflow.keras as K


def lenet5(X):
    """
    Builds a modified version of the LeNet-5
    architecture using keras

    X is a K.Input of shape (m, 28, 28, 1)
    containing the input images for the network
    m is the number of images
    """
    model = K.Sequential()
    init = K.initializers.HeNormal()

    # First convolution
    model.add(K.layers.Conv2D(filters=6,
                              kernel_size=(5, 5),
                              padding="same",
                              activation='relu',
                              kernel_initializer=init))
    # First pooling
    model.add(K.layers.MaxPooling2D(pool_size=(2, 2),
                                    strides=(2, 2)))

    # Second convolution
    model.add(K.layers.Conv2D(filters=16,
                              kernel_size=(5, 5),
                              padding="valid",
                              activation='relu',
                              kernel_initializer=init))

    # Second pooling
    model.add(K.layers.MaxPooling2D(pool_size=(2, 2),
                                    strides=(2, 2)))

    # Flatten
    model.add(K.layers.Flatten())

    # Fully connected layer 1
    model.add(K.layers.Dense(units=120,
                             activation='relu',
                             kernel_initializer=init))

    # Fully connected layer 2
    model.add(K.layers.Dense(units=84,
                             activation='relu',
                             kernel_initializer=init))

    # Fully connected layer 3 predictions
    model.add(K.layers.Dense(units=10,
                             kernel_initializer=init))

    # Fully softmax
    model.add(K.layers.Activation(activation='softmax'))

    # Optimizer
    optim = K.optimizers.Adam()

    model.compile(loss='categorical_crossentropy',
                  optimizer=optim,
                  metrics=[K.metrics.CategoricalAccuracy()])

    return model
