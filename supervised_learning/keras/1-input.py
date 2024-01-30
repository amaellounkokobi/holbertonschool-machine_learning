#!/usr/bin/env python3
"""
This module contains :
A function that builds a neural network with the Keras library

Function:
   def build_model(nx, layers, activations, lambtha, keep_prob):
"""
import tensorflow.keras as K


def build_model(nx, layers, activations, lambtha, keep_prob):
    """
    Function that builds a neural network with the Keras library

    Args:
       nx: is the number of input features to the network

       layers: is a list containing the number of nodes
       in each layer of the network

       activations: is a list containing the activation
       functions used for each layer of the network

       lambtha: is the L2 regularization parameter

       keep_prob: is the probability that a node will be
       kept for dropout

    Returns:
       the keras model
    """
    inputs = K.Input(shape=(nx,))
    K.regularizers.L2(l2=lambtha)

    L = len(layers)
    x = inputs

    for layer_unit, activation in zip(layers[:-1], activations[:-1]):
        x = K.layers.Dense(layer_unit,
                           kernel_regularizer="L2",
                           activation=activation)(x)

        x = K.layers.Dropout(rate=1 - keep_prob)(x)

    outputs = K.layers.Dense(
        layers[L - 1],
        kernel_regularizer="L2",
        activation=activations[L - 1])(x)

    model = K.Model(inputs=inputs, outputs=outputs)

    return model
