#!/usr/bin/env python3
"""
This module contains :
A function that initializes cluster centroids for K-means

Function:
def initialize(X, k):
"""
import numpy as np


def initialize(X, k):
    """
    Initializes cluster centroids for K-means
    the points are initialized randomly

    Args:
    X is a numpy.ndarray of shape (n, d) containing
    the dataset that will be used for K-means clustering

    n is the number of data points

    d is the number of dimensions for each data point

    k is a positive integer containing the number of clusters

    Returns:
    a numpy.ndarray of shape (k, d) containing the initialized
    centroids for each cluster, or None on failure

    """
    centroids = []

    # Get the lowest point coordinate
    min_point = np.min(X, axis=0)

    # Get the highest point coordinate
    max_point = np.max(X, axis=0)
    
    # Generate points with multivariate uniform random distribution
    centroids = np.random.uniform(min_point, max_point, size=(k, X.shape[1]))

    return centroids
