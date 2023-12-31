#!/usr/bin/env python3
"""
This module contains:
- A class That represents a poisson distribution

Example:
   >>> np.random.seed(0)
   ... data = np.random.poisson(5., 100).tolist()
   ... p1 = Poisson(data)
   ... print('Lambtha:', p1.lambtha)
   ... p2 = Poisson(lambtha=5)
   ... print('Lambtha:', p2.lambtha)
   Lambtha: 4.84
   Lambtha: 5.0

"""


class Poisson():
    """
    This class represent a Poisson distribution

    Attributs:
       data(list): the list of data for the distribution
       lambtha(float):the average  occurance of the event for a given time

    """
    def __init__(self, data=None, lambtha=1.):
        """
        Initializing datas and lambtha.

        Args:
           data(list): the list of data for the distribution
           lambtha(float):the average  occurance of the event for a given time

        """
        self.data = data
        if self.__data is None:
            self.lambtha = lambtha
        else:
            self.__lambtha = sum(self.__data) / len(self.__data)    

    @property
    def lambtha(self):
        """
        Getting the lambtha:
        This property calculates and returns the lambda
        of the class.

        Returns:
           (float) the value of lambda

        Raises:
           ValueError:data must be a list

        """
        return self.__lambtha


    @lambtha.setter
    def lambtha(self, value):
        """
        Setting the lambda
        this method sets the private property lambtha

        Args:
           value(float): A positif float value

        Raise:
          ValueError: Lambtha must be positive

        """
        if value <= 0:
            raise ValueError('lambtha must be a positive value')

        self.__lambtha = float(value)

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

    def pmf(self, k):
        """
        This method calculate the probability mass fuction
        for a give success

        Args:
           k(int): the number of successes
        """

        if self.data:
            if k > len(self.data) or k < 0:
                return 0

        e_val = 2.7182818285
        k_fact = self.fact(int(k))

        return (pow(e_val, -self.lambtha) * pow(self.lambtha, int(k))) / k_fact

    def cdf(self, k):
        """
        This method calculate the cumulative distribution fuction
        for a give success

        Args:
           k(int): the number of successes
        """
        result_cdf = 0

        if self.data:
            if k > len(self.data) or k < 0:
                return 0

        for x_val in range(int(k) + 1):
            result_cdf += self.pmf(x_val)

        return result_cdf

    def fact(self, value):
        """
        This method calculate the factoriel
        of a number

        Args:
           value(int):Value of factoriel
        """
        result = 1

        while value >= 1:
            result *= value
            value -= 1

        return result
