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
import pickle
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
                activation = np.matmul(
                    self.__weights['W1'], X) + self.__weights['b1']

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

        c_A_n = 'A{}'.format(self.__L)
        curr_A = cache[c_A_n]

        dZ_curr = curr_A - Y

        for l_n in range(self.__L, 0, -1):

            p_A_n = 'A{}'.format(l_n - 1)
            prev_A = cache[p_A_n]

            dW_curr = 1 / N * np.matmul(dZ_curr, prev_A.T)
            dB_curr = 1 / N * np.sum(dZ_curr, axis=1, keepdims=True)

            c_W_n = 'W{}'.format(l_n)
            curr_W = self.__weights[c_W_n]

            next_dSig = prev_A * (1 - prev_A)

            dZ_curr = np.matmul(curr_W.T, dZ_curr) * next_dSig

            c_b_n = 'b{}'.format(l_n)

            self.__weights[c_W_n] = self.__weights[c_W_n] - alpha * dW_curr
            self.__weights[c_b_n] = self.__weights[c_b_n] - alpha * dB_curr

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

    def train(self, X, Y, iterations=5000, alpha=0.05, verbose=True, graph=True, step=100):
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
        if type(step) is not int:
            raise TypeError('step must be an integer')
       # if step <= 0 or step > iterations:
       #     raise ValueError('step must be positive and <= iterations')

        x_iterations = []
        y_cost = []

        """Iteration at 0"""
        A, cache = self.forward_prop(X)
        self.gradient_descent(Y, cache, alpha)
        cost = self.cost(Y, A)
        x_iterations.append(0)
        y_cost.append(cost)

        if verbose is True:
            print('Cost after 0 iterations: {cost}'.format(cost =  self.cost(Y, A)))

        """Iteration at n"""
        for ite in range(1, iterations + 1):
            A, cache = self.forward_prop(X)
            self.gradient_descent(Y, cache, alpha)
            cost = self.cost(Y, A)

            if ite % step == 0 or ite == iterations :
                x_iterations.append(ite)
                y_cost.append(cost)
                if verbose is True:
                    print('Cost after {iteration} iterations: {cost}'.format(iteration = ite, cost = cost))

        if graph is True:
            plt.plot(x_iterations, y_cost, color='skyblue')
            plt.xlabel('iteration')
            plt.ylabel('cost')
            plt.title('Training Cost')
            plt.show()

        return np.where(A >= 0.5, 1, 0), cost

    def save(self, filename):
        """
        Saves the instance object to a file in pickle format

        Args:
           filename: is the file to which the object should be saved
        """

        if filename.find('pkl') == -1:
            filename = "{}.pkl".format(filename)

        fileObject = open(filename,'wb')

        pickle.dump(self, fileObject)
        fileObject.close()

    def load(filename):
        """
        Loads a pickled DeepNeuralNetwork object

        Args:
        filename is the file from which the object should be loaded
        """
        try:
            file_object = open(filename,'rb')
        except IOError:
            return None

        return  pickle.load(file_object)
