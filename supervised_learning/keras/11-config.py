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
    model_json = network.to_json()
    with open(filename, "w") as json_file:
        json_file.write(model_json)
    
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
    try:
        json_file = open(filename, 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        model = K.models.model_from_json(loaded_model_json)

        return model
    except IOError:
        print("Error: File does not appear to exist.")
        return 0
