#!/usr/bin/env python3
"""
This module contains :
A function that builds an inception block

Function:
   def inception_block(A_prev, filters):
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
