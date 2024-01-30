#!/usr/bin/env python3

"""
This module contains :
A function that trains a model using mini-batch gradient descent

Function:
def train_model(network,
                data,
                labels,
                batch_size,
                epochs,
                validation_data=None,
                early_stopping=False,
                patience=0,
                learning_rate_decay=False,
                alpha=0.1,
                decay_rate=1,
                save_best=False,
                filepath=None,
                verbose=True,
                shuffle=False):
"""
import tensorflow.keras as K


def train_model(network,
                data,
                labels,
                batch_size,
                epochs,
                validation_data=None,
                early_stopping=False,
                patience=0,
                learning_rate_decay=False,
                alpha=0.1,
                decay_rate=1,
                save_best=False,
                filepath=None,
                verbose=True,
                shuffle=False):
    """
    Function that trains a model using mini-batch gradient descent

    Args:
       network: is the model to train

       data: is a numpy.ndarray of shape (m, nx)
       containing the input data

       labels: is a one-hot numpy.ndarray of shape (m, classes)
       containing the labels of data

       batch_size: is the size of the batch used for
       mini-batch gradient descent

       epochs: is the number of passes through data for
       mini-batch gradient descent

       verbose: is a boolean that determines if output
       should be printed during training

       validation_data is the data to validate the model
       with, if not None

       early_stopping: is a boolean that indicates whether
       performed if validation_data exists
       based on validation loss

       learning_rate_decay: is a boolean that indicates
       whether learning rate decay should be used

       alpha: is the initial learning rate
       decay_rate: is the decay rate

       shuffle: is a boolean that determines whether to shuffle
       the batches every epoch. Normally, it is a good idea
       to shuffle, but for reproducibility, we have chosen
       to set the default to False.

       save_best: is a boolean indicating whether to save the
       model after each epoch if it is the best

       filepath: is the file path where the model should be saved

    Returns:
       the History object generated after training the model
    """
    def schedule(epoch, lr):
        return alpha / (1 + decay_rate * epoch)

    callbacks = []

    if early_stopping is True:
        early_stopping_cb = K.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=patience,
        )
        callbacks.append(early_stopping_cb)

    if learning_rate_decay is True:
        learning_decay_cb = K.callbacks.LearningRateScheduler(
            schedule,
            verbose=1)
        callbacks.append(learning_decay_cb)

    if save_best is True:
        save_best_cb = K.callbacks.ModelCheckpoint(
            filepath,
            monitor="val_loss",
            save_best_only=True)
        callbacks.append(save_best_cb)

    if validation_data is not None:
        history = network.fit(
            x=data,
            y=labels,
            batch_size=batch_size,
            epochs=epochs,
            verbose=verbose,
            shuffle=shuffle,
            callbacks=callbacks,
            validation_data=validation_data,
        )
    else:
        history = network.fit(
            x=data,
            y=labels,
            batch_size=batch_size,
            epochs=epochs,
            verbose=verbose,
            shuffle=shuffle,
            validation_data=validation_data,
        )

    return history
