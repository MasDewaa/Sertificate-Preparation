# ========================================================================================
# PROBLEM B3
#
# Build a CNN based classifier for Rock-Paper-Scissors dataset.
# Your input layer should accept 150x150 with 3 bytes color as the input shape.
# This is unlabeled data, use ImageDataGenerator to automatically label it.
# Don't use lambda layers in your model.
#
# The dataset used in this problem is created by Laurence Moroney (laurencemoroney.com).
#
# Desired accuracy AND validation_accuracy > 83%
# ========================================================================================

import urllib.request
import zipfile
import tensorflow as tf
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def solution_B3():
    # data_url = 'https://github.com/dicodingacademy/assets/releases/download/release-rps/rps.zip'
    # urllib.request.urlretrieve(data_url, 'rps.zip')
    # local_file = 'rps.zip'
    # zip_ref = zipfile.ZipFile(local_file, 'r')
    # zip_ref.extractall('data/')
    # zip_ref.close()

    TRAINING_DIR = "data/rps/"
    training_datagen = ImageDataGenerator(
        # YOUR CODE HERE
        rescale=1. / 255,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        validation_split=0.2
    )
    validation_datagen = ImageDataGenerator(rescale=1. / 255, validation_split=0.2)

    # YOUR IMAGE SIZE SHOULD BE 150x150
    # Make sure you used "categorical"
    # YOUR CODE HERE
    train_generator= training_datagen.flow_from_directory(
        TRAINING_DIR,
        target_size=(150,150),
        batch_size=32,
        class_mode='categorical',
        subset='training'
    )
    validation_generator = validation_datagen.flow_from_directory(
        TRAINING_DIR,
        target_size=(150,150),
        batch_size=32,
        class_mode='categorical',
        subset='validation'
    )

    class myCallback(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            accuracy = logs.get('accuracy')
            val_accuracy = logs.get('val_accuracy')

            if accuracy is not None and val_accuracy is not None:
                if accuracy >= 0.83 and val_accuracy >= 0.83:
                    print(
                        f"\nAccuracy: {accuracy * 100:.2f}%, Val_accuracy: {val_accuracy * 100:.2f}%, stopping training...")
                    self.model.stop_training = True

    callback = myCallback()

    model=tf.keras.models.Sequential([
    # YOUR CODE HERE, end with 3 Neuron Dense, activated by softmax
        tf.keras.layers.Conv2D(64, (3,3), activation='relu', input_shape=(150,150,3)),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Conv2D(64, (3,3),activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dense(3, activation='softmax')
    ])

    model.compile(
        loss='categorical_crossentropy',
        optimizer=tf.keras.optimizers.Adam(),
        metrics=['accuracy']
    )

    model.fit(
        train_generator,
        epochs=25,
        steps_per_epoch=len(train_generator),
        validation_data=validation_generator,
        validation_steps=len(validation_generator),
        verbose=1,
        callbacks=[callback]
    )


    return model


# The code below is to save your model as a .h5 file.
# It will be saved automatically in your Submission folder.
if __name__ == '__main__':
    # DO NOT CHANGE THIS CODE
    model=solution_B3()
    model.save("model_B3.h5")
