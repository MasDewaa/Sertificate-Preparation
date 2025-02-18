# =============================================================================
# PROBLEM B2
#
# Build a classifier for the Fashion MNIST dataset.
# The test will expect it to classify 10 classes.
# The input shape should be 28x28 monochrome. Do not resize the data.
# Your input layer should accept (28, 28) as the input shape.
#
# Don't use lambda layers in your model.
#
# Desired accuracy AND validation_accuracy > 83%
# =============================================================================
import keras.callbacks
import tensorflow as tf
import numpy as np
from keras.src.optimizers import Adam

class MyCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        accuracy = logs.get('accuracy')
        val_accuracy = logs.get('val_accuracy')

        # Cek apakah akurasi dan val_accuracy lebih dari 83%
        if accuracy is not None and val_accuracy is not None:
            if accuracy >= 0.90 and val_accuracy >= 0.90:
                print(f"\nAccuracy: {accuracy * 100:.2f}%, Val_accuracy: {val_accuracy * 100:.2f}% - Stopping training...")
                self.model.stop_training = True  # Stop training jika memenuhi ketentuan

def solution_B2():
    fashion_mnist = tf.keras.datasets.fashion_mnist
    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

    # NORMALIZE YOUR IMAGE HERE
    train_images = train_images / 255.0
    test_images = test_images / 255.0
    # DEFINE YOUR MODEL HERE
    # End with 10 Neuron Dense, activated by softmax
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    # COMPILE MODEL HERE
    model.compile(optimizer=Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    # TRAIN YOUR MODEL HERE
    # Callback untuk menghentikan pelatihan saat akurasi lebih dari 83%
    callback = MyCallback()

    # Latih model dengan callback
    model.fit(
        train_images, train_labels,
        validation_data=(test_images, test_labels),
        epochs=20,
        batch_size=64,
        verbose=2,
        callbacks=[callback]  # Menambahkan custom callback
    )

    return model


# The code below is to save your model as a .h5 file.
# It will be saved automatically in your Submission folder.
if __name__ == '__main__':
    # DO NOT CHANGE THIS CODE
    model = solution_B2()
    model.save("model_B2.h5")
