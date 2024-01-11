#!/usr/bin/env python3
"""
This module contains :
A class that defines a Deep neural network performing
binary classification

Class:
   DeepNeuralNetwork

Import:
   Numpy Library

"""
import numpy as np


class DeepNeuralNetwork():
    """
    This class defines a Deep neural network

    Attributs:


    Raises:

    """

    def __init__(self, nx, layers):
        """
        This constructor initialise class attributes:

        Args:
           nx:The number of input features to the neuron

           layers:A list representing the number of nodes
           in each layer of the network

        Raise:
           TypeError with the exception: nx must be an integer

           ValueError with the exception: nx must be a positive integer

           TypeError with the exception: layers must
           be a list of positive integers

        """
        layer_positif = np.vectorize(self.layer_positif)

        if type(nx) is not int:
            raise TypeError('nx must be an integer')
        if nx < 1:
            raise ValueError('nx must be a positive integer')
        if type(layers) is not list or not layers:
            raise TypeError('layers must be a list of positive integers')
        if all(layer_positif(layers)) is False:
            raise TypeError('layers must be a list of positive integers')

        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {}

        for l_n, layer in enumerate(layers):
            if l_n == 0:
                w_ini = np.random.randn(layers[l_n], nx) * np.sqrt(2 / nx)
                self.__weights['W{0}'.format(l_n + 1)] = w_ini
                self.__weights['b{0}'.format(l_n + 1)] = np.zeros((layer, 1))
            else:
                w_ini = np.random.randn(
                    layers[l_n], layers[l_n - 1]) * np.sqrt(
                        2 / layers[l_n - 1])
                self.__weights['W{0}'.format(l_n + 1)] = w_ini
                self.__weights['b{0}'.format(l_n + 1)] = np.zeros((layer, 1))

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

        self.__cache['A0'] = X
        for l_n in range(self.__L):
            if l_n == 0:
                activation = np.add(np.matmul(
                    self.__weights['W1'], X), self.__weights['b1'])

                A = 1/(1 + np.exp(-activation))
                self.__cache['A{0}'.format(l_n + 1)] = A
            else:
                n_W = 'W{0}'.format(l_n + 1)
                n_b = 'b{0}'.format(l_n + 1)
                n_A = 'A{0}'.format(l_n)

                val_W = self.__weights[n_W]
                val_b = self.__weights[n_b]
                val_A = self.__cache[n_A]

                activation = np.matmul(val_W, val_A) + val_b

                A = 1/(1 + np.exp(-activation))
                self.__cache['A{0}'.format(l_n + 1)] = A

        return self.__cache['A{0}'.format(self.__L)], self.__cache

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
        A, _ = self.forward_prop(X)
        cost = self.cost(Y, A)

        return np.where(A >= 0.5, 1, 0), cost

    def gradient_descent(self, Y, cache, alpha=0.05):
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
        N = Y.shape[1]

        for l_n in reversed(range(0,self.__L)):
            if l_n == self.__L - 1:
                Wn = 'W{0}'.format(l_n + 1)
                bn = 'b{0}'.format(l_n + 1)
                An = cache['A{0}'.format(l_n + 1)]
                A1n = cache['A{0}'.format(l_n)]

                dWn = 1 / N * np.matmul((An - Y), A1n.T)
                dbn = 1 / N * np.sum((An - Y), axis=1, keepdims=True)

                self.__weights[Wn] = self.__weights[Wn] - alpha * dWn
                self.__weights[bn] = self.__weights[bn] - alpha * dbn

            elif l_n > 0:
                Wn = 'W{0}'.format(l_n + 2)
                bn = 'b{0}'.format(l_n + 1)
                Wn2 = self.__weights[Wn]
                bn1 = self.__weights[bn]
                An2 = cache['A{0}'.format(l_n + 2)]
                An1 = cache['A{0}'.format(l_n + 1)]

                dA1 = An1 * (1 - An1)
                dZn1 = np.matmul(Wn2.T, (An2 - Y)) * dA1
                dWn1 = 1 / N * np.matmul(dZn1, An1.T)
                dbn1 = 1 / N * np.sum(dZn1, axis=1, keepdims=True)

                self.__weights[Wn] = self.__weights[Wn] - alpha * dWn1
                self.__weights[bn] = self.__weights[bn] - alpha * dbn1

            else:
                W2 = self.__weights['W2']
                b1 = self.__weights['b1']
                A2 = cache['A2']
                A1 = cache['A1']

                dA1 = A1 * (1 - A1)
                dZ1 = np.matmul(W2.T, (A2 - Y)) * dA1
                dW1 = 1 / N * np.matmul(dZ1, cache['A0'].T)
                db1 = 1 / N * np.sum(dZ1, axis=1, keepdims=True)

                self.__weights['W1'] = self.__weights['W1'] - alpha * dW1
                self.__weights['b1'] = self.__weights['b1'] - alpha * db1

    @property
    def L(self):
        """
        Getting the number of layers in the neural network.
        """

        return self.__L

    @property
    def cache(self):
        """
        Getting dictionary to hold all intermediary values of the network.
        """

        return self.__cache

    @property
    def weights(self):
        """
        Getting the weights.
        """

        return self.__weights

    def layer_positif(self, value):
        """
        Test if a value is positif
        """
        if value < 0:
            return False
        return True
