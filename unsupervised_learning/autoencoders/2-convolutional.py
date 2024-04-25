#!/usr/bin/env python3
"""
This module contains :
A function that creates a convolutional autoencoder

Function:
def autoencoder(input_dims, filters, latent_dims):
"""
import tensorflow.keras as K


def autoencoder(input_dims, filters, latent_dims):
    """
    Creates an autoencoder

    Args:
       input_dims (int): containing the dimensions
       of the model input

       filters:the number of filters for each
       convolutional layer in the encoder, respectively


       latent_dims: is an integer containing
       the dimensions of the latent space representation

    Returns:
       encoder is the encoder model
       decoder is the decoder model
       auto is the full autoencoder model
    """
    # *--------------*
    # | Endoder part |
    # *--------------*
    # Add an input layer of dim input_dims
    enco_in = K.Input(shape=input_dims)
    enco = enco_in

    # Add hidden layers from left to right according
    # to hidden_layers (Relu activation)
    for f in filters:
        enco = K.layers.Conv2D(filters=f,
                               kernel_size=(3, 3),
                               activation='relu',
                               padding='same')(enco)
        enco = K.layers.MaxPooling2D((2, 2),
                                     padding='same')(enco)

    # Add the lattent space layer with latent_dims
    lt_sp = K.layers.Conv2D(filters=latent_dims[-1],
                            kernel_size=(3, 3),
                            activation='relu',
                            padding='same')(enco)

    encoder = K.Model(enco_in, lt_sp)

    # *--------------*
    # | Decoder part |
    # *--------------*
    # Add an input layer of dim input_dims
    deco_in = K.Input(latent_dims)
    deco = deco_in

    # Add hidden layers from right to left according
    # to hidden_layers
    for f in reversed(filters[:-1]):
        deco = K.layers.Conv2D(filters=f,
                               kernel_size=(3, 3),
                               activation='relu',
                               padding='same')(deco)
        deco = K.layers.UpSampling2D((2, 2))(deco)

    deco = K.layers.Conv2D(filters=filters[-1],
                           kernel_size=(3, 3),
                           activation='relu',
                           padding='valid')(deco)

    deco = K.layers.UpSampling2D((2, 2))(deco)

    # Add an output layer of dim input_dims
    # (Sigmoid activation)

    out_deco = K.layers.Conv2D(filters=input_dims[-1],
                               kernel_size=(3, 3),
                               padding="same",
                               activation='sigmoid')(deco)
    decoder = K.Model(deco_in, out_deco)

    # *--------------*
    # | AutoEncoder  |
    # *--------------*
    auto_in = K.Input(input_dims)
    encode = encoder(auto_in)
    decode = decoder(encode)
    auto = K.Model(auto_in, decode)

    # Compile the model with binary cross antropy
    # and adam optimizer

    adam_opt = K.optimizers.Adam()
    auto.compile(loss='binary_crossentropy',
                 optimizer=adam_opt)

    auto.summary()

    return encoder, decoder, auto
