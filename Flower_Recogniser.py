import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing import image

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











