"""
Convolution applies small filters to an image to detect specific patterns 
like edges or textures. The output of each filter is a feature map, showing where 
and how strongly that pattern appears. Multiple filters create multiple feature maps,
capturing different aspects of the image.

In neural networks, a dense layer (also called a fully connected layer) is a layer
 where every neuron is connected to every neuron in the previous layer.
It's mainly used at the end of CNNs to make decisions or predictions

Sequential model is the simplest way to build a neural network in Keras.
It means you stack layers one after another in order

32: Number of filters (feature detectors) — the layer will learn 32 different 
filters. (3, 3): Size of each filter — a 3x3 grid of pixels.

ReLU (Rectified Linear Unit) turns negative values to zero and keeps positive 
values, helping the model learn nonlinear patterns.

128x128 — good balance for many beginner to intermediate projects.
224x224 — standard size used in many famous pretrained models (like VGG, ResNet).

Instead of using 5x5 or 7x7 once, stacking two or three 3x3 layers gives the model
more non-linearity and deeper feature extraction with fewer parameters

MaxPooling2D reduces spatial dimensions (width and height) by half, which:
Makes the model faster and less prone to overfitting.
Helps capture important features by keeping the strongest activations.

A 3D feature map is what you get when a convolutional layer produces
 multiple 2D feature maps stacked together.
"""


import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split


train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest',
)


val_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)


# 80% of data
train_generator = train_datagen.flow_from_directory(
    'Flowers', 
    target_size=(128, 128),
    batch_size=32,
    class_mode='categorical',
    subset='training'  
)

# 20% of data
validation_generator = val_datagen.flow_from_directory(
    'Flowers',
    target_size=(128, 128),
    batch_size=32,
    class_mode='categorical', # When you have more than two classes
    subset='validation' 
)


model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    layers.MaxPooling2D(2, 2),

    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),

    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),

    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(train_generator.num_classes, activation='softmax')  # Output layer
])


model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])


test_loss, test_acc = model.evaluate(test_generator)
print(f"Test accuracy: {test_acc:.2f}")


history = model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=10
)









