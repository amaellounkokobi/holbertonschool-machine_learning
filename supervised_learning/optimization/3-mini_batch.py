#!/usr/bin/env python3
"""
This module contains
A function that trains a loaded neural network
model using mini-batch gradient descent:

Function:
    def train_mini_batch(X_train,
              Y_train,
              X_valid,
              Y_valid,
              batch_size=32,
              epochs=5,
              load_path="/tmp/model.ckpt",
              save_path="/tmp/model.ckpt"):
"""
import tensorflow.compat.v1 as tf
shuffle_data = __import__('2-shuffle_data').shuffle_data


def train_mini_batch(X_train,
                     Y_train,
                     X_valid,
                     Y_valid,
                     batch_size=32,
                     epochs=5,
                     load_path="/tmp/model.ckpt",
                     save_path="/tmp/model.ckpt"):
    """
    Function that trains a loaded neural network
    model using mini-batch gradient descent:

    Args:
       X_train: is a numpy.ndarray of shape (m, 784)
       containing the training data
       m is the number of data points
       84 is the number of input features

       Y_train: is a one-hot numpy.ndarray of shape
       (m, 10) containing the training labels
       10 is the number of classes the model should classify

       X_valid: is a numpy.ndarray of shape (m, 784)
       containing the validation data

       Y_valid: is a one-hot numpy.ndarray of shape
       (m, 10) containing the validation labels

       batch_size: is the number of data points in a batch

       epochs: is the number of times the training should
       pass through the whole dataset

       load_path: is the path from which to load the model

       save_path: is the path to where the model should be saved after training

    Returns:
       the path where the model was saved

    """
    with tf.Session() as sess:
        saver = tf.train.import_meta_graph('./graph.ckpt.meta')
        saver.restore(sess, './graph.ckpt')
        x = tf.get_collection('x')[0]
        y = tf.get_collection('y')[0]
        accuracy = tf.get_collection('accuracy')[0]
        loss = tf.get_collection('loss')[0]
        train_op = tf.get_collection('train_op')[0]

        for i in range(epochs + 1):
            X_shuffle, Y_shuffle = shuffle_data(X_train, Y_train)

            train_cost = sess.run(loss, feed_dict={x: X_train, y: Y_train})
            train_accuracy = sess.run(
                accuracy, feed_dict={x: X_train, y: Y_train})
            valid_cost = sess.run(loss, feed_dict={x: X_valid, y: Y_valid})
            valid_accuracy = sess.run(
                accuracy, feed_dict={x: X_valid, y: Y_valid})

            print("After {} epochs:".format(i))
            print("\tTraining Cost: {} ".format(train_cost))
            print("\tTraining Accuracy: {}".format(train_accuracy))
            print("\tValidation Cost: {}".format(valid_cost))
            print("\tValidation Accuracy: {}".format(valid_accuracy))

            for j in range(0, X_train.shape[0], batch_size):
                X_batch = X_shuffle[j: j + batch_size]
                Y_batch = Y_shuffle[j: j + batch_size]

                if (j / batch_size) % 100 == 0 and (j / batch_size) > 0 :

                    step_cost = sess.run(
                        loss, feed_dict={x: X_batch, y: Y_batch})
                    step_accuracy = sess.run(
                        accuracy, feed_dict={x: X_batch, y: Y_batch})

                    print("\tStep {}:".format(int(j / batch_size)))
                    print("\t\tCost: {}".format(step_cost))
                    print("\t\tAccuracy: {}".format(step_accuracy))

                sess.run(train_op, feed_dict={x: X_batch, y: Y_batch})

            if X_train.shape[0] / batch_size != 0:
                X_batch = X_shuffle[j:]
                Y_batch = Y_shuffle[j:]

                sess.run(train_op, feed_dict={x: X_batch, y: Y_batch})
        saver.save(sess, save_path)

    return save_path
