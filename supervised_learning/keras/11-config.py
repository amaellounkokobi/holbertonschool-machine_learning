#!/usr/bin/env python3
"""
This module contains :
A function that saves a model’s configuration in JSON format
A function that Loads loads a model with a specific configuration

Function:
   def save_config(network, filename):
   def load_config(filename):
"""
import tensorflow.keras as K


def save_config(network, filename):
    """
    Saves a model’s configuration in JSON format

    Args:
       network: is the model to save
       filename: is the path of the file that 
       the configuration should be saved to

    Returns:
       None
    """
    K.saving.serialize_keras_object(network)
    network.save(filename)
    return None


def load_config(filename):
    """
    Loads a model with a specific configuration

    Args:
      filename: is the path of the file 
      containing the model’s configuration in JSON format

    Returns:
       The loaded model
    """
    
    model = K.models.load_model(filename)
    K.saving.deserialize_keras_object(model.get_config())

    return model
