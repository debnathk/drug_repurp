# import torch
# from torch.autograd import Variable
# import torch.nn.functional as F
# from torch.utils import data
# from torch.utils.data import SequentialSampler
# from torch import nn 
# from tqdm import tqdm
# import matplotlib.pyplot as plt
# import pandas as pd
# from time import time
# from sklearn.metrics import mean_squared_error, roc_auc_score, average_precision_score, f1_score, log_loss
# from lifelines.utils import concordance_index
# from scipy.stats import pearsonr
# import pickle 
# torch.manual_seed(2)
# import copy
# from prettytable import PrettyTable
# import os

import tensorflow as tf
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Input, Lambda, RepeatVector, GRU, TimeDistributed, Reshape
from tensorflow.keras import Model
import numpy as np
import zinc_grammar as G
np.random.seed(3)
from DeepPurpose.utils_tensorflow import *  

class CNN(Model):
    def __init__(self, encoding, **config):
        super(CNN, self).__init__()
        self.encoding = encoding
        if encoding == 'protein':
            in_ch = [26] + config['cnn_target_filters']
            kernels = config['cnn_target_kernels']
            layer_size = len(config['cnn_target_filters'])
            self.conv_layers = [Conv1D(filters=in_ch[i+1], kernel_size=kernels[i], padding='same', activation='relu', dtype='float64')
                                 for i in range(layer_size)]
            self.max_pool = MaxPooling1D(pool_size=1000)
            self.flatten = Flatten()
            self.fc = Dense(units=config['hidden_dim_protein'], activation='linear', dtype='float64')

    def call(self, v):
        x = v
        for conv_layer in self.conv_layers:
            x = conv_layer(x)
        x = self.max_pool(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x

# helper variables in Keras format for parsing the grammar
masks_K      = tf.Variable(G.masks)
ind_of_ind_K = tf.Variable(G.ind_of_ind)

MAX_LEN = 277
DIM = G.D

class gVAE():

    autoencoder = None
    
    def create(self,
               charset,
               max_length = MAX_LEN,
               latent_rep_size = 2,
               weights_file = None):
        charset_length = len(charset)
        
        x = Input(shape=(max_length, charset_length))
        _, z = self._buildEncoder(x, latent_rep_size, max_length)
        self.encoder = tf.keras.Model(x, z)

        encoded_input = Input(shape=(latent_rep_size,))
        self.decoder = tf.keras.Model(
            encoded_input,
            self._buildDecoder(
                encoded_input,
                latent_rep_size,
                max_length,
                charset_length
            )
        )

        x1 = Input(shape=(max_length, charset_length))
        vae_loss, z1 = self._buildEncoder(x1, latent_rep_size, max_length)
        self.autoencoder = tf.keras.Model(
            x1,
            self._buildDecoder(
                z1,
                latent_rep_size,
                max_length,
                charset_length
            )
        )

        # for obtaining mean and log variance of encoding distribution
        x2 = Input(shape=(max_length, charset_length))
        z_m, z_l_v = self._encoderMeanVar(x2, latent_rep_size, max_length)
        self.encoderMV = tf.keras.Model(inputs=x2, outputs=[z_m, z_l_v])

        if weights_file:
            self.autoencoder.load_weights(weights_file)
            self.encoder = tf.keras.models.model_from_config(self.encoder.get_config())
            self.decoder = tf.keras.models.model_from_config(self.decoder.get_config())
            self.encoder.load_weights(weights_file, by_name=True)
            self.decoder.load_weights(weights_file, by_name=True)
            self.encoderMV = tf.keras.models.model_from_config(self.encoderMV.get_config())
            self.encoderMV.load_weights(weights_file, by_name=True)

        self.autoencoder.compile(optimizer='Adam',
                                 loss=vae_loss,
                                 metrics=['accuracy'])

    def _encoderMeanVar(self, x, latent_rep_size, max_length, epsilon_std=0.01):
        x = Reshape((max_length, 1, DIM))(x)  # Add singleton dimension
        h = Conv1D(9, 9, activation='relu', name='conv_1')(x)
        h = Conv1D(9, 9, activation='relu', name='conv_2')(h)
        h = Conv1D(10, 11, activation='relu', name='conv_3')(h)
        h = Flatten(name='flatten_11')(h)
        h = Dense(435, activation='relu', name='dense_1')(h)

        z_mean = Dense(latent_rep_size, name='z_mean', activation='linear')(h)
        z_log_var = Dense(latent_rep_size, name='z_log_var', activation='linear')(h)

        return z_mean, z_log_var

    def _buildEncoder(self, x, latent_rep_size, max_length, epsilon_std=0.01):
        x = Reshape((max_length, 1, DIM))(x) 
        h = Conv1D(9, 9, activation='relu', name='conv_1')(x)
        h = Conv1D(9, 9, activation='relu', name='conv_2')(h)
        h = Conv1D(10, 11, activation='relu', name='conv_3')(h)
        h = Flatten(name='flatten_11')(h)
        h = Dense(435, activation='relu', name='dense_1')(h)

        z_mean = Dense(latent_rep_size, name='z_mean', activation='linear')(h)
        z_log_var = Dense(latent_rep_size, name='z_log_var', activation='linear')(h)

        def sampling(args):
            z_mean_, z_log_var_ = args
            batch_size = tf.shape(z_mean_)[0]
            epsilon = tf.random.normal(shape=(batch_size, latent_rep_size), mean=0., stddev=epsilon_std)
            return z_mean_ + tf.exp(z_log_var_ / 2) * epsilon

        def conditional(x_true, x_pred):
            most_likely = tf.argmax(x_true, axis=-1)
            ix2 = tf.expand_dims(tf.gather_nd(ind_of_ind_K, tf.expand_dims(most_likely, -1)), 1)
            ix2 = tf.cast(ix2, tf.int32)
            M2 = tf.gather_nd(masks_K, ix2)
            M3 = tf.reshape(M2, [-1, max_length, DIM])
            P2 = tf.multiply(tf.exp(x_pred), M3)
            P2 = tf.divide(P2, tf.reduce_sum(P2, axis=-1, keepdims=True))
            return P2

        def vae_loss(x, x_decoded_mean):
            x_decoded_mean = conditional(x, x_decoded_mean)
            x = tf.reshape(x, [-1])
            x_decoded_mean = tf.reshape(x_decoded_mean, [-1])
            xent_loss = max_length * tf.keras.losses.binary_crossentropy(x, x_decoded_mean)
            kl_loss = -0.5 * tf.reduce_mean(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), axis=-1)
            return xent_loss + kl_loss

        return vae_loss, Lambda(sampling, output_shape=(latent_rep_size,), name='lambda')([z_mean, z_log_var])

    def _buildDecoder(self, z, latent_rep_size, max_length, charset_length):
        h = Dense(latent_rep_size, name='latent_input', activation='relu')(z)
        h = RepeatVector(max_length, name='repeat_vector')(h)
        h = GRU(501, return_sequences=True, name='gru_1')(h)
        h = GRU(501, return_sequences=True, name='gru_2')(h)
        h = GRU(501, return_sequences=True, name='gru_3')(h)
        return TimeDistributed(Dense(charset_length), name='decoded_mean')(h)

    def save(self, filename):
        self.autoencoder.save_weights(filename)
    
    def load(self, charset, weights_file, latent_rep_size=2, max_length=MAX_LEN):
        self.create(charset, max_length=max_length, weights_file=weights_file, latent_rep_size=latent_rep_size)