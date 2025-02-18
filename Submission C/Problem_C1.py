# =============================================================================
# PROBLEM C1
#
# Given two arrays, train a neural network model to match the X to the Y.
# Predict the model with new values of X [-2.0, 10.0]
# We provide the model prediction, do not change the code.
#
# The test infrastructure expects a trained model that accepts
# an input shape of [1]
# Do not use lambda layers in your model.
#
# Please be aware that this is a linear model.
# We will test your model with values in a range as defined in the array to make sure your model is linear.
#
# Desired loss (MSE) < 1e-4
# =============================================================================

import numpy as np
import tensorflow as tf
from tensorflow import keras

class MyEpochCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        # Ambil nilai loss (MSE) dari logs
        loss = logs.get('loss')

        # Cetak loss di setiap akhir epoch
        print(f"Epoch {epoch + 1} - Loss: {loss:.6f}")

        # Jika loss sudah mencapai threshold yang diinginkan (MSE < 1e-4), stop pelatihan
        if loss < 1e-8:
            print(f"\nStopping training early, loss < 1e-4 reached at epoch {epoch + 1}.")
            self.model.stop_training = True

def solution_C1():
    # DO NOT CHANGE THIS CODE
    X = np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0], dtype=float)
    Y = np.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5], dtype=float)

    # YOUR CODE HERE
    model = tf.keras.Sequential(
        tf.keras.layers.Dense(units=1,input_shape=[1])
    )
    callback = MyEpochCallback()
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.5), loss='mse')
    model.fit(X,Y, epochs=500, verbose=1, callbacks=[callback])

    loss = model.evaluate(X, Y, verbose=1)
    print(f"Final Loss: {loss}")

    print(model.predict([-2.0, 10.0]))
    return model


# The code below is to save your model as a .h5 file
# It will be saved automatically in your Submission folder.
if __name__ == '__main__':
    # DO NOT CHANGE THIS CODE
    model = solution_C1()
    model.save("model_C1.h5")
