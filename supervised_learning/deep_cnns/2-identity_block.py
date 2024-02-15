#!/usr/bin/env python3
"""
This module contains :
A function that builds an identity block

Function:
   def identity_block(A_prev, filters):
"""
import tensorflow.keras as K


def identity_block(A_prev, filters):
    """
    Builds an identity block as described

    Args:
    A_prev is the output from the previous layer
    filters is a tuple or list containing F11, F3, F12, respectively:
       F11 is the number of filters in the first 1x1 convolution
       F3 is the number of filters in the 3x3 convolution
       F12 is the number of filters in the second 1x1 convolution
    """
    # Init Filters
    F11, F3, F12 = filters

    # 1x1 convolution
    conv_1_1 = K.layers.Conv2D(F11, (1, 1), padding="same")(A_prev)

    # Batch Norm
    Bn1 = K.layers.BatchNormalization()(conv_1_1)

    # Relu
    A_ReLu = K.layers.ReLU()(Bn1)

    # 3x3 convolution
    conv_3_3 = K.layers.Conv2D(F3, (3, 3), padding="same")(A_ReLu)

    # Batch Norm
    Bn2 = K.layers.BatchNormalization()(conv_3_3)

    # Relu
    A_ReLu_1 = K.layers.ReLU()(Bn2)

    # 1x1 convolution
    conv_1_1_2 = K.layers.Conv2D(F12, (1, 1), padding="same")(A_ReLu_1)

    # Batch Norm
    Bn3 = K.layers.BatchNormalization()(conv_1_1_2)

    # Concat input/output
    output = K.layers.Add()([A_prev, Bn3])

    # Relu
    A_ReLu_2 = K.layers.ReLU()(output)
    
    return A_ReLu_2
