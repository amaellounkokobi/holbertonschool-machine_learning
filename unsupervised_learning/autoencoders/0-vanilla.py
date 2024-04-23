#!/usr/bin/env python3
"""
This module contains :
A function that creates an autoencoder

Function:
def autoencoder(input_dims, hidden_layers, latent_dims):
"""
import tensorflow.keras as K


def autoencoder(input_dims, hidden_layers, latent_dims):
    """
    Creates an autoencoder

    Args:
       input_dims (int): containing the dimensions
       of the model input

       hidden_layers(int[]): is a list containing
       the number of nodes for each hidden layer

    Returns:
       encoder is the encoder model
       decoder is the decoder model
       auto is the full autoencoder model
    """
    hl_len = len(hidden_layers)

    # *--------------*
    # | Endoder part |
    # *--------------*
    # Add an input layer of dim input_dims
    enco_in = K.Input(shape=(input_dims,))
    enco = enco_in

    # Add hidden layers from left to right according
    # to hidden_layers (Relu activation)
    for index in range(hl_len):
        enco = K.layers.Dense(hidden_layers[index],
                              activation='relu')(enco)

    # Add the lattent space layer with latent_dims
    lt_sp = K.layers.Dense(latent_dims,
                           activation='relu')(enco)
    encoder = K.Model(enco_in, lt_sp)

    # *--------------*
    # | Decoder part |
    # *--------------*
    # Add an input layer of dim input_dims
    deco_in = K.Input(shape=(latent_dims,))
    deco = deco_in

    # Add hidden layers from right to left according
    # to hidden_layers
    for index in range(hl_len - 1, -1, -1):
        deco = K.layers.Dense(hidden_layers[index],
                              activation='relu')(deco)

    # Add an output layer of dim input_dims
    # (Sigmoid activation)
    out_deco = K.layers.Dense(input_dims,
                              activation='sigmoid')(deco)
    decoder = K.Model(deco_in, out_deco)

    # *--------------*
    # | AutoEncoder  |
    # *--------------*
    auto_in = K.Input(shape=(input_dims,))
    encode = encoder(auto_in)
    decode = decoder(encode)
    auto = K.Model(auto_in, decode)

    # Compile the model with binary cross antropy
    # and adam optimizer

    adam_opt = K.optimizers.Adam()
    auto.compile(loss='binary_crossentropy',
                 optimizer=adam_opt)

    return encoder, decoder, auto
