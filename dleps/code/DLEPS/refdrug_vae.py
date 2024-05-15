import os

os.environ["KERAS_BACKEND"] = "tensorflow"

import numpy as np
import tensorflow as tf
import keras
from keras import layers, models
# from tensorflow.keras import layers, models

# # Load dataset
# root = "/lustre/home/debnathk/dleps/code/DLEPS/reference_drug/"
# import h5py

# h5f = h5py.File(root + 'ssp_data_train.h5', 'r')
# ssp_train = h5f['data'][:]
# h5f = h5py.File(root + 'ssp_data_test.h5', 'r')
# ssp_test = h5f['data'][:]

# print(ssp_train.shape)
# print(ssp_test.shape)

# class Sampling(layers.Layer):
#     """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

#     def call(self, inputs):
#         z_mean, z_log_var = inputs
#         batch = tf.shape(z_mean)[0]
#         dim = tf.shape(z_mean)[1]
#         epsilon = tf.random.normal(shape=(batch, dim))
#         return z_mean + tf.exp(0.5 * z_log_var) * epsilon

# latent_dim = 56

# encoder_inputs = keras.Input(shape=(207, 3072))
# # Add dense layer
# x = layers.Conv1D(32, 3, activation="relu", strides=2, padding="same")(encoder_inputs)
# x = layers.Conv1D(64, 3, activation="relu", strides=2, padding="same")(x)
# x = layers.Flatten()(x)
# x = layers.Dense(16, activation="relu")(x)
# z_mean = layers.Dense(latent_dim, name="z_mean")(x)
# z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
# z = Sampling()([z_mean, z_log_var])
# encoder = keras.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")
# print(encoder.summary())

# # Decoder
# latent_inputs = keras.Input(shape=(latent_dim,))
# x = layers.Dense(16, activation="relu")(latent_inputs)
# x = layers.Reshape((1, 16))(x)
# x = layers.Conv1DTranspose(64, 3, activation="relu", strides=2, padding="same")(x)
# x = layers.Conv1DTranspose(32, 3, activation="relu", strides=2, padding="same")(x)
# x = layers.Flatten()(x)
# decoder_outputs = layers.Dense(207 * 3072, activation="sigmoid")(x)
# decoder_outputs = layers.Reshape((207, 3072))(decoder_outputs)
# decoder = keras.Model(latent_inputs, decoder_outputs, name="decoder")
# print(decoder.summary())

class VAE(models.Model):
    def __init__(self, encoder, decoder, **kwargs):
        super(VAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.total_loss_tracker = tf.keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = tf.keras.metrics.Mean(name="reconstruction_loss")
        self.kl_loss_tracker = tf.keras.metrics.Mean(name="kl_loss")

    def train_step(self, data):
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data)
            reconstruction = self.decoder(z)
            reconstruction_loss = tf.reduce_mean(tf.keras.losses.binary_crossentropy(data, reconstruction))
            kl_loss = -0.5 * tf.reduce_mean(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            total_loss = reconstruction_loss + kl_loss

        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)

        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }

if __name__ == "__main__":

    # Instantiate the VAE model
    vae = VAE(encoder=encoder, decoder=decoder)

    # Compile the model
    vae.compile(optimizer=tf.keras.optimizers.Adam())
    num_epochs, batch_size = 3, 32

    import matplotlib.pyplot as plt

    # Assuming you've already trained the VAE using the code you provided

    # Train the VAE model
    history = vae.fit(ssp_train, epochs=num_epochs, batch_size=batch_size)

    # Save model weights
    vae.save_weights(root + 'refdrug_vae.h5')

    # Plot the losses over epochs
    def plot_losses(history):
        loss = history.history['loss']
        reconstruction_loss = history.history['reconstruction_loss']
        kl_loss = history.history['kl_loss']

        epochs = range(1, len(loss) + 1)

        plt.figure(figsize=(10, 5))
        plt.plot(epochs, loss, label='Total Loss')
        plt.plot(epochs, reconstruction_loss, label='Reconstruction Loss')
        plt.plot(epochs, kl_loss, label='KL Divergence Loss')

        plt.title('Training Losses')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(root + 'training_losses.png')
        plt.show()

    # Plot the losses
    plot_losses(history)


