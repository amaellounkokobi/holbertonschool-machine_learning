#!/usr/bin/env python3
"""
This module contains :
A function that makes a prediction using a neural network

Function:
   def predict(network, data, verbose=False):
"""
import tensorflow.keras as K


def predict(network, data, verbose=False):
    """
    Function that makes a prediction using a neural network

    Args:
       network: is the network model to make the prediction with

       data: is the input data to make the prediction with

       verbose: is a boolean that determines if output
       should be printed during the prediction process

       Returns:
          The prediction for the data
    """
    prediction = network.predict(data, verbose=verbose)

    return prediction
