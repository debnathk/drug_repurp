import numpy as np
import keras
from keras import layers

class CNN:
    def __init__(self):
        self.input_shape = (26, 1000)  # Hardcoded input shape
        self.encoder_model = self._build_encoder()

    def _build_encoder(self):
        # Hardcoded configuration
        in_channels = [26, 32, 64, 96]
        kernels = [4, 8, 12]
        output_units = 256

        # Create layers
        inputs = keras.Input(shape=self.input_shape)
        x = inputs
        for i in range(len(kernels)):
            x = layers.Conv1D(filters=in_channels[i+1], kernel_size=kernels[i], activation="relu", padding="same")(x)
            x = layers.MaxPooling1D(pool_size=2, padding="same")(x)
        x = layers.GlobalMaxPooling1D()(x)
        x = layers.Dense(output_units, activation="relu")(x)
        outputs = layers.Dense(output_units, activation="softmax")(x)

        encoder_model = keras.models.Model(inputs=inputs, outputs=outputs)

        return encoder_model
    
    def summary(self):
        return self.encoder_model.summary()

    def encode(self, data):
        return self.encoder_model.predict(data)
    
if __name__ == "__main__":

    # Instantiate the CNNEncoder class
    cnn_encoder = CNN()
    print(cnn_encoder.summary())

    # Generate some synthetic data
    num_samples = 10
    x_data = np.random.rand(num_samples, 26, 1000)

    # Encode the data
    encoded_data = cnn_encoder.encode(x_data)

    print("Encoded data shape:", encoded_data.shape)
