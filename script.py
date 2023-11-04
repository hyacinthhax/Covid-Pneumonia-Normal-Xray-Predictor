# System/Py Imports
import os

# Standard TF imports
import numpy as np
import tensorflow as tf
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.metrics import AUC
from tensorflow.keras.layers import Conv2D, Dense, Flatten, MaxPooling2D, Input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


# Define Hyperparameters
learning_rate = 0.001
epochs = 5
batch_size = 32

# Define the desired image dimensions
desired_width = 256
desired_height = 256

# Define the paths to your training and validation data
train_data_dir = str(os.getcwd()) + "\\Covid19-dataset\\train\\"
val_data_dir = str(os.getcwd()) + "\\Covid19-dataset\\test\\"

data_generator = ImageDataGenerator()

# Create data iterators
train_generator = data_generator.flow_from_directory(
    train_data_dir,
    target_size=(desired_width, desired_height),  # Resize the images to your desired dimensions
    batch_size=batch_size,
    class_mode='categorical',  # Change to 'binary' if you have only two classes
)

validation_generator = data_generator.flow_from_directory(
    val_data_dir,
    target_size=(desired_width, desired_height),
    batch_size=batch_size,
    class_mode='categorical',
)

model = tf.keras.Sequential()
model.add(tf.keras.Input(shape=(256, 256, 3)))
model.add(Conv2D(8, 3, activation="relu", padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(8, 3, activation="relu", padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(16, activation="relu"))
model.add(Dense(3, activation="softmax"))

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss=CategoricalCrossentropy(), metrics=[AUC()])

# Trains using training_generator and validation_generator
history = model.fit(
    train_generator,
    steps_per_epoch=len(train_generator),
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=len(validation_generator))

# Plot the training and validation loss
plt.figure()
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig('datasetRun.png')

model.save('pneumoniaCovidTester.keras')
