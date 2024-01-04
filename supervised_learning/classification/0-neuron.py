#!/usr/bin/env python3
"""
This module contains :
A class that defines a single neuron performing
binary classification

Class:
   Neuron

Import:
   Numpy Library

"""
import numpy as np


class Neuron():
    """
    This class defines a single neuron

    Attributs:
       W: The weights vector for the neuron.
       b: The bias for the neuron.
       A: The activated output of the neuron (prediction).

    Raises:
       TypeError with the exception: nx must be an integer
       ValueError with the exception: nx must be a positive integer
    """

    def __init__(self, nx):
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

        self.__W = np.random.normal(size=(1,nx))
        self.__b = 0
        self.__A = 0

    @property
    def W(self):
        """
        Getting the weight vector
        """

        return self.__W

    @W.setter
    def W(self,value):
        """
        Setting the weight vector
        """

        self.__W = value

    @property
    def b(self):
        """
        Getting the bias
        """

        return self.__b

    @b.setter
    def b(self,value):
        """
        Setting the bias
        """

        self.__b = value

    @property
    def A(self):
        """
        Getting the activation output
        """

        return self.__A

    @A.setter
    def A(self,value):
        """
        Setting the activation output
        """

        self.__A = value
