#!/usr/bin/env python3
"""
This module contains :
A function that saves a model’s weights
A function that loads a model’s weights

Function:
   def save_model(network, filename):
   def load_model(filename):
"""
import tensorflow.keras as K


def save_weights(network, filename, save_format='h5'):
    """
    Saves a model’s weights

    Args:
       network: is the model to save

       filename: is the path of the file
       that the model should be saved to

       save_format is the format in which
       the weights should be saved

    Returns:
       None
    """
    if filname.find(save_format) < 0:
        filename = "{}.{}".format(filename, save_format)

    network.save_weights(filename)

    return None


def load_weights(network, filename):
    """
    Loads a model's weights.

    Args:
      network: is the model to which the
      weights should be loaded

      filename:  is the path of the file
      that the model should be loaded from

    Returns:
       The loaded model
    """
    network.load_weights(filename)

    return network
