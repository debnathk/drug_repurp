from keras.models import Model
from keras.layers import Input, Dense, Dropout, Conv1D, GlobalMaxPooling1D

class ConvNet(object):
    def __init__ (self, input_shape=(978, 2), dropout=0.2):
        self.input_shape = input_shape
        self.dropout = dropout
        self.model = self._build_model()

    def _build_model(self):
        visible_1 = Input(shape=(978, 2))
        conv1D_1 = Conv1D(128, 3, activation='relu')(visible_1)
        maxpooling_1 = GlobalMaxPooling1D()(conv1D_1)
        dense_1 = Dense(512, activation='relu')(maxpooling_1)
        drop_1 = Dropout(self.dropout)(dense_1)
        dense_2 = Dense(512, activation='relu')(drop_1)
        drop_2 = Dropout(self.dropout)(dense_2)
        dense_3 = Dense(512, activation='relu')(drop_2)
        drop_3 = Dropout(self.dropout)(dense_3)
        dense_4 = Dense(512, activation='relu')(drop_3)
        drop_4 = Dropout(self.dropout)(dense_4)
        output = Dense(1, activation='linear')(drop_4)
        model = Model(inputs=visible_1, outputs=output)
        return model

    def compile_model(self, optimizer='adam', loss='mean_squared_error', metrics=['mae']):
        self.model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    def train_model(self, X_train, y_train, epochs=100, batch_size=50, validation_data=None):
        history = self.model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=validation_data)
        return history

    def evaluate(self, X_val, y_val):
        return self.model.evaluate(X_val, y_val)

    def predict(self, X_test):
        return self.model.predict(X_test)

    def summary(self):
        self.model.summary()