#!/usr/bin/env python3
"""
This module contains :
A class that defines a neural network with one hidden layer performing
binary classification

Class:
   NeuralNetwork

Import:
   Numpy Library

"""
import numpy as np


class NeuralNetwork():
    """
    This class defines a neural network

    Attributs:
       W1: The weights vector for the hidden layer 1.
       b1: The bias for the hidden layer 1.
       A1: The activated output for the hidden layer (prediction).
       W2: The weights vector for the output neuron.
       b2: The bias for the output neuron.
       A2: The activated output of the neuron (prediction).

    Raises:
       TypeError with the exception: nx must be an integer
       ValueError with the exception: nx must be a positive integer
       TypeError with the exception: nodes must be an integer
       ValueError with the exception: nodes must be a positive integer
    """

    def __init__(self, nx, nodes):
        """
        This constructor initialise class attributes:
           Initialized using a random normal distribution

        Args:
           nx:the number of input features to the neuron
        """

        if type(nx) is not int:
            raise TypeError('nx must be an integer')
        if nx < 1:
            raise ValueError('nx must be a positive integer')
        if type(nodes) is not int:
            raise TypeError('nodes must be an integer')
        if nodes < 1:
            raise ValueError('nodes must be a positive integer')

        self.__W1 = np.random.normal(size=(nodes, nx))
        self.__b1 = np.zeros((nodes, 1))
        self.__A1 = 0

        self.__W2 = np.random.normal(size=(1, nodes))
        self.__b2 = 0
        self.__A2 = 0

    @property
    def W1(self):
        """
        Getting the hidden layer weight vector
        """

        return self.__W1

    @property
    def b1(self):
        """
        Getting the hidden layer bias
        """

        return self.__b1

    @property
    def A1(self):
        """
        Getting the hidden layer activation output
        """

        return self.__A1

    @property
    def W2(self):
        """
        Getting the output layer weight vector
        """

        return self.__W2

    @property
    def b2(self):
        """
        Getting the output layer bias
        """

        return self.__b2

    @property
    def A2(self):
        """
        Getting the output layer activation
        """

        return self.__A2

    @W1.setter
    def W1(self, value):
        """
        setting the hidden layer weight vector
        """

        self.__W1 = value

    @W2.setter
    def W2(self, value):
        """
        setting the output layer weight vector
        """

        self.__W2 = value

    @b1.setter
    def b1(self, value):
        """
        setting the hidden layer bias vector
        """

        self.__b1 = value

    @b2.setter
    def b2(self, value):
        """
        setting the output layer bias
        """

        self.__b2 = value

    @A1.setter
    def A1(self, value):
        """
        setting the hidden layer activation
        """

        self.__A1 = value

    @b2.setter
    def A2(self, value):
        """
        setting the output layer activation
        """

        self.__A2 = value
