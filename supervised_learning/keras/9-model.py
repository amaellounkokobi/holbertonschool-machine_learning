#!/usr/bin/env python3
"""
This module contains :
A function that saves an entire model.
A function that Loads an entire model.

Function:
   def save_model(network, filename):
   def load_model(filename):
"""
import tensorflow.keras as K


def save_model(network, filename):
    """
    Saves an entire model.

    Args:
       network: is the model to save
       filename: is the path of the file
       that the model should be saved to

    Returns:
       None
    """
    network.save(filename)
    return None


def load_model(filename):
    """
    Loads an entire model.

    Args:
      filename:  is the path of the file
      that the model should be loaded from

    Returns:
       The loaded model
    """
    model = K.models.load_model(filename)

    return model
