#!/usr/bin/env python3
"""                                                                                                                                                                                                        This module contains :                                                                                                                                                                                     a function that builds a neural network with the Keras library                                                                                                                                             Function:                                                                                                                                                                                                     def build_model(nx, layers, activations, lambtha, keep_prob):

"""
import tensorflow as tf
import tensorflow.keras as K


def build_model(nx, layers, activations, lambtha, keep_prob):
    """                                                                                                                                                                                                        Function that builds a neural network with the Keras library                                                                                                                                               Args:                                                                                                                                                                                                         nx: is the number of input features to the network                                                                                                                                                         layers: is a list containing the number of nodes                                                                                                                                                           in each layer of the network                                                                                                                                                                               activations: is a list containing the activation                                                                                                                                                           functions used for each layer of the network                                                                                                                                                               lambtha: is the L2 regularization parameter
       keep_prob: is the probability that a node 
       will be kept for dropout
    Returns:
       the keras model
     """
    # create keras model                                                                                                                                                                                    
    model = K.Sequential()
    K.regularizers.L2(l2=lambtha)
    model.add(K.layers.Dense(layers[0],
                             kernel_regularizer="L2",
                             activation=activations[0],
                             input_shape=(nx,)))

    for layer_unit, activation in zip(layers[1:], activations[1:]):
        model.add(K.layers.Dropout(rate=keep_prob))
        model.add(K.layers.Dense(layer_unit,
                                 kernel_regularizer="L2",
                                 activation=activation))

    model.build()
    return model
