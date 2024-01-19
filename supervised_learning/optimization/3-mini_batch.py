#!/usr/bin/env python3
"""
This module contains
A function that trains a loaded neural network model using mini-batch gradient descent:

Function:
    def train_mini_batch(X_train, Y_train, X_valid, Y_valid, batch_size=32, epochs=5, load_path="/tmp/model.ckpt", save_path="/tmp/model.ckpt"):
"""
import numpy as np


def train_mini_batch(X_train, Y_train, X_valid, Y_valid, batch_size=32, epochs=5, load_path="/tmp/model.ckpt", save_path="/tmp/model.ckpt"):
    """
    Function that trains a loaded neural network model using mini-batch gradient descent:

    Args:
       X_train: is a numpy.ndarray of shape (m, 784) containing the training data
       m is the number of data points
       84 is the number of input features

       Y_train: is a one-hot numpy.ndarray of shape (m, 10) containing the training labels
       10 is the number of classes the model should classify

       X_valid: is a numpy.ndarray of shape (m, 784) containing the validation data

       Y_valid: is a one-hot numpy.ndarray of shape (m, 10) containing the validation labels

       batch_size: is the number of data points in a batch

       epochs: is the number of times the training should pass through the whole dataset

       load_path: is the path from which to load the model

       save_path: is the path to where the model should be saved after training

    Returns:
       the path where the model was saved

    """

    return ""
