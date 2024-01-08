#!/usr/bin/env python3
"""
This module contains :
A class that defines a single neuron performing
binary classification

Class:
   Neuron

Examples:
  lib_train = np.load('../data/Binary_Train.npz')
  X_3D, Y = lib_train['X'], lib_train['Y']
  X = X_3D.reshape((X_3D.shape[0], -1)).T

  np.random.seed(0)
  neuron = Neuron(X.shape[0])
  A, cost = neuron.evaluate(X, Y)
  print(A)
  print(cost)

  #SHOW CHARACTER
  image_values = X[:,1220]  # Ajoutez les valeurs complètes ici
  assert len(image_values) == 28 * 28,
  "La taille de la liste ne correspond pas à une image 28x28."
  for i in range(28):
    row = image_values[i * 28: (i + 1) * 28]
    ascii_row = "".join("O" if value == 0 else "1" for value in row)
    print(ascii_row)

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

        self.__W = np.random.normal(size=(1, nx))
        self.__b = 0
        self.__A = 0

    @property
    def W(self):
        """
        Getting the weight vector
        """

        return self.__W

    @property
    def b(self):
        """
        Getting the bias
        """

        return self.__b

    @property
    def A(self):
        """
        Getting the activation output
        """

        return self.__A

    def forward_prop(self, X):
        """
        Calculates the forward propagation of the neuron
        Updates the private attribute __A
        The neuron should use a sigmoid activation function

        Args:
           X is a numpy.ndarray with shape (nx, m)
           that contains the input data

        Returns:
           the private attribute __A

        """
        activation = np.add(np.matmul(self.__W, X), self.__b)
        self.__A = 1/(1 + np.exp(-activation))

        return self.__A

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
        Evaluates the neuron’s predictions

        Args:
        X: is a numpy.ndarray with shape (nx, m) that contains the
        input data nx is the number of input features to the neuron
        m is the number of examples

        Y: is a numpy.ndarray with shape (1, m) that contains
        the correct labels for the input data

        """
        activation = np.add(np.matmul(self.__W, X), self.__b)
        A = 1/(1 + np.exp(-activation))
        cost = self.cost(Y, A)
        step = np.vectorize(self.step)

        return step(A, 0.5), cost

    def step(self, value, threshold):
        """
        Take a value between 1 and zeo and returns 1 or 0
        according to a threshold

        Args:
           Value: float number between 0 and 1

        Returns:
           0 or 1

        """

        if value >= threshold:
            return 1
        else:
            return 0

    def gradient_descent(self, X, Y, A, alpha=0.05):
        """
        Calculates one pass of gradient descent on the neuron
        Updates the private attributes __W and __b

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

        dW = 1 / N * np.dot((A - Y), X.T)
        db = 1 / N * np.sum((A - Y))

        self.__W = self.__W - alpha * dW
        self.__b = self.__b - alpha * db

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
        for ite in range(iterations):
            A = self.forward_prop(X)
            self.gradient_descent(X, Y, A, alpha)

        cost = self.cost(Y, A)
        return A, cost
