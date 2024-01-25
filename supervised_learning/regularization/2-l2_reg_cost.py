#!/usr/bin/env python3
"""
This module contains :
A function that calculates the cost of a neural
network with L2 regularization:


Function:
   def l2_reg_cost(cost):
"""
import tensorflow.compat.v1 as tf


def l2_reg_cost(cost):
    """
    function that calculates the cost of a neural
    network with L2 regularization:

    Args:
       cost: is a tensor containing the cost of
       the network without L2 regularization

    Returns:
       a tensor containing the cost of the
       network accounting for L2 regularization

    """

    L2_reg = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    cost += L2_reg

    return cost
