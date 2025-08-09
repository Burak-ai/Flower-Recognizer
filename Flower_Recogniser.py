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
    class_mode='categorical',
    subset='validation' 
)

