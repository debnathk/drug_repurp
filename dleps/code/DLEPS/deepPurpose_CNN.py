import numpy as np
import keras
from keras import layers

class CNN:
    def __init__(self):
        self.input_shape = (26, 1000)  # Hardcoded input shape
        self.output_units = 56
        self.encoder_model = self._build_encoder()

    def _build_encoder(self):
        inputs = keras.Input(shape=self.input_shape)
        
        # Reshape to add channel dimension (needed for Conv2D)
        x = layers.Reshape((self.input_shape[0], self.input_shape[1], 1))(inputs)
        
        # Convolutional layers
        x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
        x = layers.MaxPooling2D((2, 2), padding='same')(x)
        
        x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
        x = layers.MaxPooling2D((2, 2), padding='same')(x)
        
        # Flattening the layer
        x = layers.Flatten()(x)
        
        # Dense layers
        x = layers.Dense(128, activation='relu')(x)
        
        outputs = layers.Dense(self.output_units, activation='softmax')(x)

        encoder_model = keras.models.Model(inputs=inputs, outputs=outputs)

        return encoder_model
    
    def summary(self):
        return self.encoder_model.summary()

    def encode(self, data):
        return self.encoder_model.predict(data)
    
if __name__ == "__main__":

    # Instantiate the CNNEncoder class
    cnn_encoder = CNN()

    # Generate some synthetic data
    num_samples = 10
    x_data = np.random.rand(num_samples, 26, 1000)

    # Encode the data
    encoded_data = cnn_encoder.encode(x_data)

    print("Encoded data shape:", encoded_data.shape)
