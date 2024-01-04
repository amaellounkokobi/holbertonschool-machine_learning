#!/usr/bin/env python3

"""
This module contains:
- A class That represents an exponential distribution

Example:
>>> data = np.random.exponential(0.5, 100).tolist()
... e1 = Exponential(data)
... print('Lambtha:', e1.lambtha)
Lambtha: 2.1771114730906937

"""


class Exponential():
    """
    This class represent a Exponential distribution

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
        self.e_val = 2.7182818285
        self.data = data
        
        if self.__data is None:
            self.lambtha = lambtha
        else:
            self.__lambtha = 1 / (sum(self.__data) / len(self.__data))

    @property
    def lambtha(self):
        """
        Getting the lambtha:
        This property calculates and returns the lambtha
        of the class.

        Returns:
           (float) the value of lambtha

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

    def pdf(self, x):
        """
        This method calculate the probability mass fuction
        for a give success

        Args:
           k(int): the number of successes
        """

        if self.data:
            if x > len(self.data) or x < 0:
                return 0

        return self.lambtha * pow(self.e_val, (-self.lambtha * x))

    def cdf(self, x):
        """
        This method calculate the cumulative distribution fuction
        for a give success

        Args:
           k(int): the number of successes
        """
        if self.data:
            if x > len(self.data) or x < 0:
                return 0

        return 1 - pow(self.e_val, (-self.lambtha * x))
