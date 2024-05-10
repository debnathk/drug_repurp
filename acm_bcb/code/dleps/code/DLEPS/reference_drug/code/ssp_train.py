# Import libraries
from keras.models import Sequential
from keras.layers import Dense, Flatten, Input, Dropout
from keras.models import Model
from keras import regularizers 
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, LearningRateScheduler

# Create a fully connected network
visible_1 = Input(shape=(207, 3072))
flaten_1 = Flatten()(visible_1)
dense_11 = Dense(1024, activation='relu')(flaten_1)
drop_1 = Dropout(0.4)(dense_11)
dense_12 = Dense(512, activation='relu')(drop_1)
drop_2 = Dropout(0.4)(dense_12)
dense_13 = Dense(256, activation='relu')(drop_2)
drop_3 = Dropout(0.4)(dense_13)
dense_14 = Dense(128, activation='relu')(drop_3)
drop_4 = Dropout(0.4)(dense_14)
dense_15 = Dense(56, activation='relu')(drop_4)
output_1 = Dense(1, activation='linear')(dense_15)
sequential = Model(inputs=visible_1, outputs=output_1)
print(sequential.summary())

# Create predictor setup

# Hyperparameters
epochs = 10
batch_size = 25

# Compile the model
optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, amsgrad=False)
sequential.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])  # Mean Squared Error and Mean Absolute Error as metrics for regression

# Load dataset
import h5py

h5f = h5py.File('ssp_data_train.h5', 'r')
ssp_train = h5f['data'][:]
h5f = h5py.File('ssp_data_test.h5', 'r')
ssp_test = h5f['data'][:]
h5f2 = h5py.File('y_train.h5', 'r')
y_train = h5f2['data'][:]
h5f4 = h5py.File('y_test.h5', 'r')
y_test = h5f4['data'][:]

print(ssp_train.shape)
print(y_train.shape)
print(ssp_test.shape)
print(y_test.shape)

# Train the model

# Set up ModelCheckpoint to save weights for the epoch with the best validation loss
checkpoint_filepath = "ssp_weights.h5"
model_checkpoint = ModelCheckpoint(checkpoint_filepath, save_best_only=True, monitor='val_loss', mode='min', verbose=1)
history = sequential.fit(ssp_train, y_train, validation_data=(ssp_test, y_test), epochs=epochs, batch_size=batch_size, verbose=2, callbacks=[model_checkpoint])

# Load the best weights based on the optimal epoch
checkpoint_filepath = "ssp_weights.h5"
sequential.load_weights(checkpoint_filepath)

# Evaluate the model on the test set
results = sequential.evaluate(ssp_test, y_test, batch_size=batch_size)

# Print the evaluation results
print("Test Loss:", results[0])
print("Test MAE:", results[1])

# Find the epoch with the minimum validation loss
optimal_epoch = history.history['val_loss'].index(min(history.history['val_loss'])) + 1
print(f"Optimal Epoch: {optimal_epoch} loss: {history.history['val_loss'][optimal_epoch-1]}")

# Print metrics
from sklearn.metrics import r2_score
import numpy as np
y_pred = sequential.predict(ssp_test)
print(np.corrcoef(y_test, y_pred.ravel()))
print(r2_score(y_test, y_pred))