import os
import pdb
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" 
os.environ["CUDA_VISIBLE_DEVICES"] = "4"

# from densenet import DLEPS
from dleps_predictor2_playground import DLEPS
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
import h5py

# pdb.set_trace()
dleps_p = DLEPS()
model = dleps_p.model[0]
print(model.summary())


h5f = h5py.File('../../data/vae_train.h5', 'r')
vae_train = h5f['data'][:]
h5f = h5py.File('../../data/vae_test.h5', 'r')
vae_test = h5f['data'][:]
h5f = h5py.File('../../data/gene_exp_data_train.h5', 'r')
seq_train = h5f['data'][:]
h5f = h5py.File('../../data/gene_exp_data_test.h5', 'r')
seq_test = h5f['data'][:]
h5f2 = h5py.File('../../data/y_train.h5', 'r')
y_train = h5f2['data'][:]
h5f2 = h5py.File('../../data/y_test.h5', 'r')
y_test = h5f2['data'][:]
h5f3 = h5py.File('reference_drug/data/ssp_data_train.h5', 'r')
refdrug_train = h5f3['data'][:]
h5f4 = h5py.File('reference_drug/data/ssp_data_test.h5', 'r')
refdrug_test = h5f4['data'][:]

print(vae_train.shape)
print(vae_test.shape)
print(seq_train.shape)
print(seq_test.shape)
print(refdrug_train.shape)
print(refdrug_test.shape)
print(y_train.shape)
print(y_test.shape)

# compile the model
optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, amsgrad=False)
model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['mae']) 

# Use ModelCheckpoint to save model and weights
from keras.callbacks import ModelCheckpoint
# filepath = "weights.best.sequential.hdf5" - best
# filepath = "weights.best.sample.hdf5"
filepath = "all_test.weights.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')

# Use the final model to get a single output
epochs = 1000
batch_size = 50
early_stopping = EarlyStopping(monitor='val_mae', patience=10)
history = model.fit([seq_train, vae_train, refdrug_train], y_train, batch_size=batch_size, epochs=epochs, callbacks=[checkpoint, early_stopping], validation_data=([seq_test, vae_test, refdrug_test], y_test))

model.load_weights("all_test.weights.hdf5")

# Evaluate the model on the test set
results = model.evaluate([seq_test, vae_test, refdrug_test], y_test)
print("Test Loss:", results[0])
print("Test MAE:", results[1])

import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
import numpy as np
y_pred = model.predict([seq_test, vae_test])
print(np.corrcoef(y_test, y_pred.ravel()))
print(r2_score(y_test, y_pred))