#!/usr/bin/env python3
"""
This module contains :
A function that builds the inception network

Function:
   def inception_network()
"""
import tensorflow.keras as K


def inception_block(A_prev, filters):
    """
    Builds an inception block

    Args:
    A_prev is the output from the previous layer
    filters is a tuple or list containing
    F1, F3R, F3,F5R, F5, FPP, respectively:

       F1 is the number of filters in the 1x1 convolution

       F3R is the number of filters in the 1x1
       convolution before the 3x3 convolution

       F3 is the number of filters in the 3x3 convolution

       F5R is the number of filters in the 1x1 convolution
       before the 5x5 convolution

       F5 is the number of filters in the 5x5 convolution

       FPP is the number of filters in the 1x1
       convolution after the max pooling

    """
    # Get filters
    F1, F3R, F3, F5R, F5, FPP = filters

    # First step convolution 1x1
    s1 = K.layers.Conv2D(F1, (1, 1), activation='relu')(A_prev)

    # Second step Conv 1x1 reduction and conv 3x3 (padding same)
    s2 = K.layers.Conv2D(F3R, (1, 1), activation='relu')(A_prev)
    s2 = K.layers.Conv2D(F3, (3, 3), padding="same", activation='relu')(s2)

    # Third step Conv 1x1 reduction and conv 5x5 (padding same)
    s3 = K.layers.Conv2D(F5R, (1, 1), activation='relu')(A_prev)
    s3 = K.layers.Conv2D(F5, (5, 5), padding="same", activation='relu')(s3)

    # Fourth step Max pooling 3x3 and conv 1x1
    s4 = K.layers.MaxPooling2D((3, 3), strides=(1, 1), padding="same")(A_prev)
    s4 = K.layers.Conv2D(FPP, (1, 1), activation='relu')(s4)

    # Concatanate output
    output = K.layers.Concatenate()([s1, s2, s3, s4])

    return output


def inception_network():
    """
    Builds the inception network
    the input data will have shape (224, 224, 3)
    """
    # Init kernels
    init = K.initializers.HeNormal()

    # Input datas
    inputs = K.Input(shape=(224, 224, 3))

    # Conv 7x7/2
    conv7x7_2 = K.layers.Conv2D(filters=64,
                                kernel_size=(7, 7),
                                strides=(2, 2),
                                activation='relu',
                                padding='same',
                                kernel_initializer=init)(inputs)

    # Max pool 3x3/2
    max_pool = K.layers.MaxPooling2D((3, 3),
                                     strides=(2, 2),
                                     padding='same')(conv7x7_2)

    # Conv 3x3/1
    conv3x3_1 = K.layers.Conv2D(filters=64,
                                kernel_size=(1, 1),
                                activation='relu',
                                kernel_initializer=init)(max_pool)

    conv3x3_1 = K.layers.Conv2D(filters=192,
                                kernel_size=(3, 3),
                                strides=(1, 1),
                                padding="same",
                                activation='relu',
                                kernel_initializer=init)(conv3x3_1)

    # Max pool 3x3/2
    max_pool2 = K.layers.MaxPooling2D((3, 3),
                                      strides=(2, 2),
                                      padding="same")(conv3x3_1)
    # Inception bloc 3a
    i_block3a = inception_block(max_pool2, [64, 96, 128, 16, 32, 32])

    # Inception bloc 3b
    i_block3b = inception_block(i_block3a, [128, 128, 192, 32, 96, 64])

    # Max pool 3x3/2
    max_pool3 = K.layers.MaxPooling2D((3, 3),
                                      strides=(2, 2),
                                      padding="same")(i_block3b)

    # Inception bloc 4a
    i_block4a = inception_block(max_pool3, [192, 96, 208, 16, 48, 64])

    # Inception bloc 4b
    i_block4b = inception_block(i_block4a, [160, 112, 224, 24, 64, 64])

    # Inception bloc 4c
    i_block4c = inception_block(i_block4b, [128, 128, 256, 24, 64, 64])

    # Inception bloc 4d
    i_block4d = inception_block(i_block4c, [112, 144, 288, 32, 64, 64])

    # Inception bloc 4e
    i_block4e = inception_block(i_block4d, [256, 160, 320, 32, 128, 128])

    # Max pool 3x3/2
    max_pool4 = K.layers.MaxPooling2D((3, 3),
                                      strides=(2, 2),
                                      padding="same")(i_block4e)

    # Inception bloc 5a
    i_block5a = inception_block(max_pool4, [256, 160, 320, 32, 128, 128])

    # Inception bloc 5b
    i_block5b = inception_block(i_block5a, [384, 192, 384, 48, 128, 128])

    # Avg pool
    avg_pool = K.layers.AveragePooling2D((7, 7),
                                         strides=(1, 1))(i_block5b)

    # Dropout layer
    drop_out = K.layers.Dropout(0.4)(avg_pool)

    # Linear
    output = K.layers.Dense(1000,
                            activation='softmax',
                            kernel_initializer=init)(drop_out)

    network = K.Model(inputs=inputs, outputs=output)

    return network
