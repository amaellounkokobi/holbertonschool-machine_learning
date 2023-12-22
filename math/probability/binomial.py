#!/usr/bin/env python3
"""
this module contains:
- A class that represents a binomial distribution

Example:
>>> np.random.seed(0)
... data = np.random.binomial(50, 0.6, 100).tolist()
... b1 = Binomial(data)
... print('n:', b1.n, "p:", b1.p)
n: 50 p: 0.606


Classes:
   Binomial

"""


class Binomial():
    """
    This class represents a Binomial distribution

    Attributs:

    """

    def __init__(self, data=None, n=1, p=0.5):
        """
        Initializing attributs

        Args:
           data(list): the list of data for the distribution
           n(int): the Bernouilli trials
           p(float): the probability of a success
        """

        self.data = data

        if self.__data is None:
            self.n = n
            self.p = p
        else:
            self.__n, self.__p = self.calculate_n_p()

    def calculate_n_p(self):
        """
        This method calculates the probability of success

        Returns:
           (float) success probability

        """
        sum_res = 0
        len_data = len(self.__data)
        mean = sum(self.__data) / len_data

        for x_val in self.__data:
            sum_res = sum_res + pow(x_val - mean, 2)

        variance = round(sum_res / len_data)
        result_p = 1 - (variance / mean)
        q = 1 - result_p
        result_n = round(variance / (result_p * (1 - result_p)))
        result_p = variance / (result_n * q)

        return result_n, result_p

    @property
    def n(self):
        """
        Getting the Bernouilli trials

        Returns:
        (float): Number n
        """
        return self.__n

    @n.setter
    def n(self, value):
        """
        Setting the Bernouilli trials

        Args:
        (float): Positive value

        Raise:
        ValueError: n must be a positive value

        """
        if value <= 0:
            raise ValueError('n must be a positive value')

        self.__n = value

    @property
    def p(self):
        """
        Getting the probability of a success

        Returns:
           (int): Number p
        """
        return self.__p

    @p.setter
    def p(self, value):
        """
        Setting the number of succese

        Args:
           (float): Value between 0 and 1

        Raise:
           ValueError p must be greater than 0 and less than 1

        """
        if value < 0:
            raise ValueError('p must be greater than 0 and less than 1')

        if value >= 1:
            raise ValueError('p must be greater than 0 and less than 1')

        self.__p = value

    @property
    def data(self):
        """
        Getting data lists:
        This method returns the data

        Returns:
           None if the data is empty or has no data

        """
        return self.__data

    @data.setter
    def data(self, value):
        """
        Setting the data lists:
        This sets the data list that is gonna be proceed

        Raises:
           ValueError: If data does not contain at least two data points
           TypeError: If data is not a list

        """
        if value is None:
            self.__data = value
        else:
            if type(value) is not list:
                raise TypeError('data must be a list')

            if len(value) < 2:
                raise ValueError('data must contain multiple values')

            self.__data = value
