import argparse
import numpy as np
import h5py
import matplotlib.pyplot as plt
from keras.models import Sequential, Model
from keras.layers import Dense, Flatten, Input, Dropout
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras import regularizers

def train_model(epochs, batch_size, train_data, train_labels, test_data, test_labels, checkpoint_filepath, trainable=True):
    # define the keras model
    visible_1 = Input(shape=train_data.shape[1:])
    flaten_1 = Flatten()(visible_1)
    dense_11 = Dense(1024, activation='relu', kernel_regularizer=regularizers.l2(0.00001))(flaten_1)
    drop_1 = Dropout(0.4)(dense_11)
    dense_12 = Dense(512, activation='relu', kernel_regularizer=regularizers.l2(0.00001))(drop_1)
    drop_2 = Dropout(0.4)(dense_12)
    dense_13 = Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.00001))(drop_2)
    drop_3 = Dropout(0.4)(dense_13)
    dense_14 = Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.00001))(drop_3)
    drop_4 = Dropout(0.4)(dense_14)
    dense_15 = Dense(56, activation='relu', kernel_regularizer=regularizers.l2(0.00001))(drop_4)
    output_1 = Dense(1, activation='linear')(dense_15)
    sequential = Model(inputs=visible_1, outputs=output_1)

    print(sequential.summary())

    # Compile the model
    optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, amsgrad=False)
    sequential.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['mae'])  

    # Set up ModelCheckpoint to save weights for the epoch with the best validation loss
    model_checkpoint = ModelCheckpoint(checkpoint_filepath, save_best_only=True, monitor='val_loss', mode='min', verbose=1)

    # Train the model if trainable
    if trainable:
        # Train the model
        history = sequential.fit(train_data, train_labels, validation_data=(test_data, test_labels), epochs=epochs, batch_size=batch_size, verbose=2, callbacks=[model_checkpoint])

        # Load the best weights based on the optimal epoch
        sequential.load_weights(checkpoint_filepath)

        results = sequential.evaluate(test_data, test_labels, batch_size=batch_size)

        print("Test Loss:", results[0])
        print("Test MAE:", results[1])

        optimal_epoch = history.history['val_loss'].index(min(history.history['val_loss'])) + 1
        print(f"Optimal Epoch: {optimal_epoch} loss: {history.history['val_loss'][optimal_epoch-1]}")

        y_pred = sequential.predict(test_data)
        print(np.corrcoef(test_labels, y_pred.ravel()))
    else:
        # Load the saved weights
        sequential.load_weights(checkpoint_filepath)
        print("Weights loaded successfully.")

        y_pred = sequential.predict(test_data)
        print(np.corrcoef(test_labels, y_pred.ravel()))

    return sequential, history if trainable else None

def plot_results(history, output_dir):
    if history:
        # Plot training & validation loss values
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('Model loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Test'], loc='upper left')
        plt.savefig(output_dir + '/loss_plot.png')
        plt.close()

        # Plot training & validation MAE values
        plt.plot(history.history['mae'])
        plt.plot(history.history['val_mae'])
        plt.title('Model MAE')
        plt.ylabel('MAE')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Test'], loc='upper left')
        plt.savefig(output_dir + '/mae_plot.png')
        plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train and evaluate a deep learning model')
    parser.add_argument('--epochs', type=int, help='number of epochs for training', default=10)
    parser.add_argument('--batch_size', type=int, help='batch size for training', default=50)
    parser.add_argument('--train_data_path', type=str, help='path to training data', default='../../data/gene_exp_data_train.h5')
    parser.add_argument('--train_labels_path', type=str, help='path to training labels', default='../../data/y_train.h5')
    parser.add_argument('--test_data_path', type=str, help='path to test data', default='../../data/gene_exp_data_test.h5')
    parser.add_argument('--test_labels_path', type=str, help='path to test labels', default='../../data/y_test.h5')
    parser.add_argument('--checkpoint_filepath', type=str, help='path to save model checkpoints', default='test_weights.h5')
    parser.add_argument('--output_dir', type=str, help='directory to save plot images', default='plots')
    parser.add_argument('--trainable', type=bool, help='whether to train the model or just load weights', default=True)
    args = parser.parse_args()

    # Load data
    with h5py.File(args.train_data_path, 'r') as h5f:
        train_data = h5f['data'][:]
    with h5py.File(args.train_labels_path, 'r') as h5f2:
        train_labels = h5f2['data'][:]
    with h5py.File(args.test_data_path, 'r') as h5f3:
        test_data = h5f3['data'][:]
    with h5py.File(args.test_labels_path, 'r') as h5f4:
        test_labels = h5f4['data'][:]

    # Train the model
    trained_model, history = train_model(args.epochs, args.batch_size, train_data, train_labels, test_data, test_labels, args.checkpoint_filepath)

    # Plot results
    plot_results(history, args.output_dir)
