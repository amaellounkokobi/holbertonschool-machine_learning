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

    def forward_prop(self, X):
        """
        Calculates the forward propagation of the neural network
        Updates the private attribute A1 A2
        The neuron should use a sigmoid activation function

        Args:
           X is a numpy.ndarray with shape (nx, m)
           that contains the input data

        Returns:
           the private attribute A1, A2

        """
        activation1 = np.add(np.matmul(self.__W1, X), self.__b1)
        self.__A1 = 1/(1 + np.exp(-activation1))

        activation2 = np.add(np.matmul(self.__W2, self.__A1), self.__b2)
        self.__A2 = 1/(1 + np.exp(-activation2))

        return self.__A1, self.__A2

    def cost(self, Y, A):
        """
        Calculates the cost of the model using logistic regression

        Args:
           Y: is a numpy.ndarray with shape (1, m)
           that contains the correct labels for the input data

           A: is a numpy.ndarray with shape (1, m)
           containing the activated output of the neuron for each example

        Returns:

           The cost
        """
        N = Y.shape[1]
        cost_function = -(Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A))

        return 1 / N * np.sum(cost_function)

    def evaluate(self, X, Y):
        """
        Evaluates the neural networf’s predictions

        Args:
        X: is a numpy.ndarray with shape (nx, m) that contains the
        input data nx is the number of input features to the neuron
        m is the number of examples

        Y: is a numpy.ndarray with shape (1, m) that contains
        the correct labels for the input data

        """
        _, A = self.forward_prop(X)

        cost = self.cost(Y, A)
        
        return np.where(A >= 0.5, 1, 0), cost

    def gradient_descent(self, X, Y, A1, A2, alpha=0.05):
        """
        Calculates one pass of gradient descent on the neural network
        Updates the private attributes __W1 __W2 and __b1 __b2

        Args:
        X: is a numpy.ndarray with shape (nx, m) that contains
        the input data nx is the number of input features to the neuron
        m is the number of examples

        Y: is a numpy.ndarray with shape (1, m) that contains
        the correct labels for the input data

        A: is a numpy.ndarray with shape (1, m) containing the
        activated output of the neuron for each example alpha
        is the learning rate

        """
        N = X.shape[1]

        """ Back propagation """
        dW2 = 1 / N * np.matmul((A2 - Y), A1.T)
        db2 = 1 / N * np.sum((A2 - Y))

        dA1 = A1 * (1 - A1)
        dZ1 = np.matmul(self.__W2.T, (A2 - Y)) * dA1

        dW1 = 1 / N * np.matmul(dZ1, X.T)
        db1 = 1 / N * np.sum(dZ1)

        """ Update parameters """
        self.__W2 = self.__W2 - alpha * dW2
        self.__b2 = self.__b2 - alpha * db2

        self.__W1 = self.__W1 - alpha * dW1
        self.__b1 = self.__b1 - alpha * db1
        print("W1",(self.__W1 - alpha * dW1)[0][0])
        print("W1-",(self.__W1[0][0]))

    def train(self, X, Y, iterations=5000, alpha=0.05):
        """
        Trains the neuron
         X: is a numpy.ndarray with shape (nx, m) that contains
        the input data nx is the number of input features to the neuron
        m is the number of examples

        Y: is a numpy.ndarray with shape (1, m) that contains
        the correct labels for the input data

        iterations: is the number of iterations to train over

        alpha: is the learning rate
        """

        if type(iterations) is not int:
            raise TypeError('iterations must be an integer')
        if iterations < 0:
            raise ValueError('iterations must be a positive integer')
        if type(alpha) is not float:
            raise TypeError('alpha must be a float')
        if alpha < 0:
            raise ValueError('alpha must be positive')

        for ite in range(iterations):
            A1, A2 = self.forward_prop(X)
            cost = self.cost(Y, A2)
            self.gradient_descent(X, Y, A1, A2, alpha)

        return self.evaluate(X, Y)
