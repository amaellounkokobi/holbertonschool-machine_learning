#!/usr/bin/env python3
"""
This module contains a function that builds,
trains, and saves a neural network classifier


Function:
   def train(X_train,
             Y_train,
             X_valid,
             Y_valid,
             layer_sizes,
             activations,
             alpha,
             iterations,
             save_path="/tmp/model.ckpt"):
"""
import tensorflow.compat.v1 as tf
calculate_accuracy = __import__('3-calculate_accuracy').calculate_accuracy
calculate_loss = __import__('4-calculate_loss').calculate_loss
create_placeholders = __import__('0-create_placeholders').create_placeholders
create_train_op = __import__('5-create_train_op').create_train_op
forward_prop = __import__('2-forward_prop').forward_prop


def train(X_train,
          Y_train,
          X_valid,
          Y_valid,
          layer_sizes,
          activations,
          alpha,
          iterations,
          save_path="/tmp/model.ckpt"):
    """
    Function that builds, trains, and saves a neural network classifier

    Args:
       X_train: is a numpy.ndarray containing the training input data
       Y_train: is a numpy.ndarray containing the training labels
       X_valid: is a numpy.ndarray containing the validation input data
       Y_valid: is a numpy.ndarray containing the validation labels
       layer_sizes: is a list containing the number of nodes in
       each layer of the network
       activations: is a list containing the activation functions
       for each layer of the network
       alpha: is the learning rate
       iterations: is the number of iterations to train over
       save_path: designates where to save the model

    """
    nx = X_train.shape[1]
    classes = Y_train.shape[1]
    x, y = create_placeholders(nx, classes)
    y_pred = forward_prop(x, layer_sizes, activations)
    loss = calculate_loss(y, y_pred)
    accuracy = calculate_accuracy(y, y_pred)
    train_op = create_train_op(loss, alpha)
    saver = tf.train.Saver()
    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)

        # First train
        valid_cost = sess.run(
            loss, feed_dict={x: X_valid, y: Y_valid})
        valid_accuracy = sess.run(
            accuracy, feed_dict={x: X_valid, y: Y_valid})
        train_cost = sess.run(
            loss, feed_dict={x: X_train, y: Y_train})
        train_accuracy = sess.run(
            accuracy, feed_dict={x: X_train, y: Y_train})

        print("After {} epochs:".format(0))
        print("\tTraining Cost: {} ".format(train_cost))
        print("\tTraining Accuracy: {}".format(train_accuracy))
        print("\tValidation Cost: {}".format(valid_cost))
        print("\tValidation Accuracy: {}".format(valid_accuracy))

        for i in range(iterations):

            if i % 100 == 0 and i > 0:
                valid_cost = sess.run(
                    loss, feed_dict={x: X_valid, y: Y_valid})
                valid_accuracy = sess.run(
                    accuracy, feed_dict={x: X_valid, y: Y_valid})
                train_cost = sess.run(
                    loss, feed_dict={x: X_train, y: Y_train})
                train_accuracy = sess.run(
                    accuracy, feed_dict={x: X_train, y: Y_train})

                print("After {} epochs:".format(i))
                print("\tTraining Cost: {} ".format(train_cost))
                print("\tTraining Accuracy: {}".format(train_accuracy))
                print("\tValidation Cost: {}".format(valid_cost))
                print("\tValidation Accuracy: {}".format(valid_accuracy))

            sess.run(train_op, feed_dict={x: X_train, y: Y_train})

        # Last train
        valid_cost = sess.run(
            loss, feed_dict={x: X_valid, y: Y_valid})
        valid_accuracy = sess.run(
            accuracy, feed_dict={x: X_valid, y: Y_valid})
        train_cost = sess.run(
            loss, feed_dict={x: X_train, y: Y_train})
        train_accuracy = sess.run(
            accuracy, feed_dict={x: X_train, y: Y_train})

        print("After {} epochs:".format(i + 1))
        print("\tTraining Cost: {} ".format(train_cost))
        print("\tTraining Accuracy: {}".format(train_accuracy))
        print("\tValidation Cost: {}".format(valid_cost))
        print("\tValidation Accuracy: {}".format(valid_accuracy))

        save_path = saver.save(sess, save_path)

    return save_path
