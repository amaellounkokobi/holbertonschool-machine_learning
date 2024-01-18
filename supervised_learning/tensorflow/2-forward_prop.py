#!/usr/bin/env python3
"""
This module contains:
A function that creates the forward propagation graph 
for the neural network:
Function:
   def forward_prop(x, layer_sizes=[], activations=[]):
"""
import tensorflow.compat.v1 as tf
create_layer = __import__('1-create_layer').create_layer


def forward_prop(x, layer_sizes=[], activations=[]):
    """
    Forward propagation graph for the neural network

    Args:
        x: is the placeholder for the input data

        layer_sizes: is a list containing the number of nodes in each 
        layer of the network

        activations: is a list containing the activation functions for 
        each layer of the network

        Returns: the prediction of the network in tensor form
    """
    prev = x
    for nodes, activation in zip(layer_sizes, activations):
        prev = create_layer(prev, nodes, activation)

    return prev
