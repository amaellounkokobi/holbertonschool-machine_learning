#!/usr/bin/env python3
"""
This module contains a script that performs
transfer learning and classify 10 classes of cifar10 dataset
using imagenet weight

Function:
    def preprocess_data(X, Y):
"""
import tensorflow.keras as K
import matplotlib.pyplot as plt


def preprocess_data(X, Y):
    """
    Preprocessing datas
    Args:
        X: data train
        Y: correct predictions
    Returns:
        X_p : Normalize Data
        Y_p : One hot label
    """    
    # Convert to One hot
    Y_p = K.utils.to_categorical(Y)
    # Normalize
    X_p = X / 255
    
    return X_p, Y_p

# Load the Dataset
(X_train, Y_train), (X_valid, Y_valid) = K.datasets.cifar10.load_data()

# Adapt data to model NASNET
X_prep , Y_prep = preprocess_data(X_train, Y_train)
X_vp , Y_vp = preprocess_data(X_valid, Y_valid)

# Choose base_model
base_model = K.applications.MobileNet(
    input_shape=(224,224,3),
    alpha=1.0,
    depth_multiplier=1,
    dropout=0.001,
    include_top=False,
    weights="imagenet",
    input_tensor=None,
    pooling=None,
    classes=1000,
    classifier_activation="softmax",
)

# Prepare resize layer 
resize = lambda x: K.preprocessing.image.smart_resize(x, (224 , 224))
resize_layer =  K.layers.Lambda(resize)

# New model
inputs = K.Input(shape=(32,32,3))

# Resize
x = resize_layer(inputs)

# Data Augmentation
x = K.layers.RandomFlip("horizontal")(x)
x = K.layers.RandomRotation(0.2)(x)
x = K.layers.RandomTranslation(height_factor=0.1, width_factor=0.1)(x)
x = K.layers.RandomContrast(factor=0.1)(x)
x = base_model(x, training=True)
x = K.layers.MaxPooling2D()(x)
x = K.layers.Flatten()(x)
x = K.layers.Dense(10, activation='softmax')(x)
model = K.Model(inputs=inputs, outputs=x)

# Compile model
model.compile(optimizer='SGD',
              loss='categorical_crossentropy',
              metrics=[K.metrics.CategoricalAccuracy(name="accuracy")])

# Model fit
history = model.fit(x=X_prep,
          y=Y_prep,
          batch_size=16,
          epochs=9,
          verbose=True,
          shuffle=True,
          validation_data=(X_vp, Y_vp))

# Save model
model.save("./cifar10.h5")

# Plotting accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title("Metrics")
plt.ylabel("accuracy")
plt.xlabel("epoch")
plt.legend(['Train', 'Val'], loc='upper left')
plt.show()

# Plotting loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title("losses")
plt.xlabel("epoch")
plt.ylabel("loss")
plt.legend(['Train','Val'],loc='upper left')
plt.show()
