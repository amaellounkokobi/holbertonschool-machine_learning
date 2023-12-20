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
        self.lambtha = lambtha

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
        if self.__data is None:
            return float(self.__lambtha)

        gen_lambtha = sum(self.__data) / len(self.__data)

        return gen_lambtha

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

        self.__lambtha = value

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
