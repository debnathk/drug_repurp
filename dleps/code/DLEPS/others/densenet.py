from keras.models import Model
from keras.layers import Dense, Input, Flatten, Dropout

class DenseNet:
    def __init__(self, input_shape=(978, 2), hidden_units=[1024, 512, 256, 128, 56], dropout=0.2):
        self.input_shape = input_shape
        self.hidden_units = hidden_units
        self.dropout = dropout
        self.model = self.build_model()

    def build_model(self):
        visible_1 = Input(shape=(978, 2))
        flatten_1 = Flatten()(visible_1)
        x = flatten_1
        for unit in self.hidden_units:
            x = Dense(unit, activation='relu')(x)
            x = Dropout(self.dropout)(x)
        output_1 = Dense(1, activation='linear')(x)
        model = Model(inputs=visible_1, outputs=output_1)

        return model

    def compile_model(self, optimizer='adam', loss='mean_squared_error', metrics=['mae']):
        self.model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    def train_model(self, X_train, y_train, epochs=100, batch_size=50, validation_data=None):
        history = self.model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=validation_data)
        return history

    def evaluate_model(self, X_test, y_test):
        return self.model.evaluate(X_test, y_test)

    def predict(self, X):
        return self.model.predict(X)
    
    # def save_weights(self, weight_file):
    #     return self.model.save_weights(weight_file)
    
    # def load_weights(self, weight_file):
    #     return self.model.load_weights(weight_file='../../data/sequential.h5')

    def summary(self):
        self.model.summary()

# Example usage:
# input_dim and output_dim depend on your dataset
# model = DenseNet()
# model.compile_model()

# Train the model with your data
# model.train_model(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_val, y_val))

# Save pretrained weights
# model.save_weights('pretrained_weights.h5')

# Create a new instance of the model and load pretrained weights
# new_model = DenseNet()
# new_model.compile_model()
# new_model.load_weights('pretrained_weights.h5')

# Now you can use new_model for predictions or further training with pretrained weights
# predictions = new_model.predict(X_new_data)




