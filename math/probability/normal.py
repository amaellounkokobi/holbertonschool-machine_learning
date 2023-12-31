#!/usr/bin/env python3
"""
This module contains:
- A class That represents a normal distribution

Example:
   >>> data = np.random.normal(70, 10, 100).tolist()
   ... n1 = Normal(data)
   ... print('Z(90):', n1.z_score(90))
   ... print('X(2):', n1.x_value(2))
   Z(90): 1.9250185174272068
   X(2): 90.75572504967644

   >>> n2 = Normal(mean=70, stddev=10)
   ... print()
   ... print('Z(90):', n2.z_score(90))
   ... print('X(2):', n2.x_value(2))
   Z(90): 2.0
   X(2): 90.0

Classes:
   Normal

"""


class Normal():
    """
    This class represent a Normal distribution

    Attributs:
       data(list): the list of data for the distribution
       mean(float): the mean of the distribution
       stddev(float): is the standard deviation of the distribution

    """
    def __init__(self, data=None, mean=0., stddev=1.):
        """
        Initializing attributs.

        Args:
           data(list): the list of data for the distribution
           mean(float):the average values of datas
           stddev(float):standard deviation

        """
        self.e_val = 2.7182818285
        self.pi_val = 3.1415926536
        self.data = data

        if self.__data is None:
            self.mean = mean
            self.stddev = stddev
        else:
            self.__mean = sum(self.__data) / len(self.__data)
            self.__stddev = self.calculate_stddev()

    @property
    def mean(self):
        """
        Getting the mean of the distribution:
        This property calculates and returns the mean
        of the distribution.

        Returns:
           (float) the value of mean
        """
        return self.__mean

    @mean.setter
    def mean(self, value):
        """
        Setting the mean
        this method sets the private property mean of the distribution

        Args:
           value(float): A positif float value

        """
        self.__mean = float(value)

    @property
    def stddev(self):
        """
        Getting the stddev:

        Return:
           float standard deviation
        """
        return self.__stddev

    @stddev.setter
    def stddev(self, value):
        """
        Setting the stddev
        this method sets the private property stddev

        Args:
           value(float): A positif float value

        Raise:
          ValueError: Lambtha must be positive

        """
        if value <= 0:
            raise ValueError('stddev must be a positive value')

        self.__stddev = float(value)

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

    def calculate_stddev(self):
        """
        This method calculates the standard deviation

        Returns:
           float standard deviation.
        """
        Σ_res = 0
        n_pop = len(self.__data)

        for x_val in self.__data:
            Σ_res = Σ_res + pow(x_val - self.mean, 2)

        return pow(Σ_res / n_pop, 0.5)

    def z_score(self, x):
        """
        This method calculate the z-score of x

        Args:
           x: is the x-value
        """
        return (x - self.mean) / self.stddev

    def x_value(self, z):
        """
        This method calculates the x-value of a given z-score

        Args:
           z: is the z-score
        """
        return self.mean + (z * self.stddev)

    def pdf(self, x):
        """
        This method calculate the probability density fuction
        for a given x value

        Args:
           x(int): is the x-value
        """
        sigma = self.stddev
        mu = self.mean
        pow_xpr = -(pow(x - mu, 2)) / (2 * pow(sigma, 2))
        sqrt_xpr = pow(2 * self.pi_val * pow(sigma, 2), 0.5)

        return (1 / sqrt_xpr) * pow(self.e_val, pow_xpr)

    def erf(self, x):
        """
        This method implements an approximation
        of the error function

        Args:
           x(int): is the x-value

        """
        const = 2 / self.pi_val ** 0.5
        factor = x - (x**3) / 3 + (x**5) / 10 - (x**7) / 42 + (x**9) / 216

        return const * factor

    def cdf(self, x):
        """
        This method calculate the cumulative distribution fuction
        for a given x-value

        Args:
           x(int):
        """
        z = (x - self.__mean) / (self.__stddev * (2 ** 0.5))
        result_cdf = (1 / 2) * (1 + self.erf(z))

        return result_cdf
