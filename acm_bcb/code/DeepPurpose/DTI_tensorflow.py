import tensorflow as tf
tf.enable_eager_execution()
import keras
from keras.layers import Input, Conv1D, Flatten, Dense, Lambda, RepeatVector, GRU, TimeDistributed, concatenate, Dropout
from keras.models import Model
from keras.losses import BinaryCrossentropy, MeanSquaredError
from keras.optimizers import Adam
import numpy as np
# import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, log_loss, mean_squared_error
from lifelines.utils import concordance_index
from scipy.stats import pearsonr
import pickle
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import molecule_vae_single
# from dleps_predictor2 import DLEPS

import os

from DeepPurpose.utils_tensorflow import *   
from DeepPurpose.encoders_tensorflow import *

# import tensorflow as tf
# from tensorflow.keras.layers import Dense, Dropout, concatenate

class Classifier(tf.keras.Sequential):
    def __init__(self, model_drug, model_protein, **config):
        super(Classifier, self).__init__()
        self.input_dim_drug = config['hidden_dim_drug']
        self.input_dim_protein = config['hidden_dim_protein']

        self.model_drug = model_drug
        self.model_protein = model_protein

        self.dropout_rate = 0.1

        self.hidden_dims = config['cls_hidden_dims']
        layer_size = len(self.hidden_dims) + 1
        dims = [self.input_dim_drug + self.input_dim_protein] + self.hidden_dims + [1]
        
        self.predictor = [Dense(units=dims[i+1], activation='relu' if i != layer_size-1 else None) for i in range(layer_size)]

    def call(self, v_D, v_P):
        # each encoding
        v_D = self.model_drug.encode(v_D)
        v_P = self.model_protein(v_P)

        # tf.print("Shape of v_D:", tf.shape(v_D))
        # tf.print("Shape of v_P:", tf.shape(v_P))

        # concatenate and classify
        v_f = concatenate([v_D, v_P], axis=1)
        for i, l in enumerate(self.predictor):
            if i==(len(self.predictor)-1):
                v_f = l(v_f)
            else:
                v_f = tf.nn.relu(Dropout(rate=self.dropout_rate)(l(v_f)))
        return v_f

def model_initialize(**config):
	model = DBTA(**config)
	return model

class DBTA(object):
    def __init__(self, **config):
        self.config = config
        self.drug_encoding = config['drug_encoding']
        self.target_encoding = config['target_encoding']
        self.result_folder = config['result_folder']
        os.makedirs(self.result_folder, exist_ok=True)
        self.binary = False
        self.config['num_workers'] = self.config.get('num_workers', 0)
        self.config['decay'] = self.config.get('decay', 0)

        if self.drug_encoding == 'gVAE':
            grammar_weights = '../data/vae.hdf5'
            grammar_model = molecule_vae_single.ZincGrammarModel(grammar_weights)
            self.model_drug = grammar_model
        # Add other drug encoding methods here

        if self.target_encoding == 'CNN':
            self.model_protein = CNN('protein', **config)

        self.model = Classifier(self.model_drug, self.model_protein, **config)

    def test(self, data_generator, repurposing_mode=False, test=False):
        y_pred = []
        y_label = []
        for v_d, v_p, label in data_generator:
            score = self.model(v_d, v_p)
            if self.binary:
                logits = tf.squeeze(tf.nn.sigmoid(score), axis=1).numpy()
            else:
                logits = tf.squeeze(score, axis=1).numpy()
            label_ids = label.numpy().flatten()
            y_label.extend(label_ids.tolist())
            y_pred.extend(logits.tolist())
        y_pred = np.array(y_pred)
        y_label = np.array(y_label)

        if self.binary:
            if repurposing_mode:
                return y_pred
            if test:
                self.plot_roc_auc(y_pred, y_label)
                self.plot_pr_auc(y_pred, y_label)
            outputs = np.where(y_pred >= 0.5, 1, 0)
            return roc_auc_score(y_label, y_pred), average_precision_score(y_label, y_pred), f1_score(y_label, outputs), log_loss(y_label, outputs), y_pred
        else:
            if repurposing_mode:
                return y_pred
            return mean_squared_error(y_label, y_pred), pearsonr(y_label, y_pred)[0], pearsonr(y_label, y_pred)[1], concordance_index(y_label, y_pred), y_pred

    def train(self, train, val=None, test=None, verbose=True):
        if len(train.Label.unique()) == 2:
            self.binary = True
            self.config['binary'] = True

        lr = self.config['LR']
        decay = self.config['decay']
        batch_size = self.config['batch_size']
        train_epoch = self.config['train_epoch']
        test_every_X_epoch = self.config.get('test_every_X_epoch', 40)
        loss_history = []

        optimizer = Adam(learning_rate=lr, epsilon=1e-08)

        if verbose:
            print('--- Data Preparation ---')

        train_dataset = data_process_loader(train.index.values, train.Label.values, train, **self.config)
        # train_dataset = train_dataset.batch(batch_size, drop_remainder=False)

        if val is not None:
            val_dataset = data_process_loader(val.index.values, val.Label.values, val, **self.config)
            # val_dataset = val_dataset.batch(batch_size, drop_remainder=False)

        if test is not None:
            test_dataset = data_process_loader(test.index.values, test.Label.values, test, **self.config)
            # test_dataset = test_dataset.batch(batch_size, drop_remainder=False)

        if self.binary:
            max_auc = 0
        else:
            max_mse = 10000
        model_max = self.model

        valid_metric_record = []
        valid_metric_header = ["# epoch"]
        if self.binary:
            valid_metric_header.extend(["AUROC", "AUPRC", "F1"])
        else:
            valid_metric_header.extend(["MSE", "Pearson Correlation", "with p-value", "Concordance Index"])

        if verbose:
            print('--- Go for Training ---')

        for epoch in range(train_epoch):
            for i, (v_d, v_p, label) in enumerate(tqdm(train_dataset)):
                # print("Shape of v_d", v_d.shape)
                # print("Shape of v_p", v_p.shape)
                with tf.GradientTape() as tape:
                    score = self.model(v_d, v_p)
                    if self.binary:
                        loss_fn = BinaryCrossentropy()
                        loss = loss_fn(label, tf.squeeze(score))
                    else:
                        loss_fn = MeanSquaredError()
                        loss = loss_fn(label, tf.squeeze(score, axis=1))
                loss_history.append(loss.numpy())

                grads = tape.gradient(loss, self.model.trainable_variables)
                optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

                if verbose and i % 100 == 0:
                    print(f'Training at Epoch {epoch + 1} iteration {i} with loss {loss.numpy():.7f}')

            if val is not None:
                if self.binary:
                    auc, auprc, f1, loss, logits = self.test(val_dataset.make_one_shot_iterator(), test=False)
                    lst = [f"epoch {epoch}", auc, auprc, f1]
                    valid_metric_record.append(lst)
                    if auc > max_auc:
                        model_max = self.model
                        max_auc = auc
                    if verbose:
                        print(f'Validation at Epoch {epoch + 1}, AUROC: {auc:.7f}, AUPRC: {auprc:.7f}, F1: {f1:.7f}, Cross-entropy Loss: {loss:.7f}')
                else:
                    mse, r2, p_val, CI, logits = self.test(val_dataset.make_one_shot_iterator(), test=False)
                    lst = [f"epoch {epoch}", mse, r2, p_val, CI]
                    valid_metric_record.append(lst)
                    if mse < max_mse:
                        model_max = self.model
                        max_mse = mse
                    if verbose:
                        print(f'Validation at Epoch {epoch + 1}, MSE: {mse:.7f}, Pearson Correlation: {r2:.7f} with p-value: {p_val:.2E}, Concordance Index: {CI:.7f}')
            else:
                model_max = self.model

        self.model = model_max

        if test is not None:
            if verbose:
                print('--- Go for Testing ---')
            if self.binary:
                auc, auprc, f1, loss, logits = self.test(test_dataset.make_one_shot_iterator(), test=True)
                print(f'Testing AUROC: {auc:.7f}, AUPRC: {auprc:.7f}, F1: {f1:.7f}, Cross-entropy Loss: {loss:.7f}')
            else:
                mse, r2, p_val, CI, logits = self.test(test_dataset.make_one_shot_iterator(), test=False)
                print(f'Testing MSE: {mse:.7f}, Pearson Correlation: {r2:.7f} with p-value: {p_val:.2E}, Concordance Index: {CI:.7f}')
            np.save(os.path.join(self.result_folder, f"{self.drug_encoding}_{self.target_encoding}_logits.npy"), logits)

            # Save test results
            test_table = [["AUROC", "AUPRC", "F1"]] if self.binary else [["MSE", "Pearson Correlation", "with p-value", "Concordance Index"]]
            test_table.append([auc, auprc, f1] if self.binary else [mse, r2, p_val, CI])
            prettytable_file = os.path.join(self.result_folder, f"{self.drug_encoding}_{self.target_encoding}_test_markdowntable.txt")
            with open(prettytable_file, 'w') as fp:
                fp.write('\n'.join(['|'.join(map(str, row)) for row in test_table]))

        # Plot learning curve
        iter_num = list(range(1, len(loss_history) + 1))
        plt.figure(3)
        plt.plot(iter_num, loss_history, "bo-")
        plt.xlabel("iteration", fontsize=16)
        plt.ylabel("loss value", fontsize=16)
        pkl_file = os.path.join(self.result_folder, f"{self.drug_encoding}_{self.target_encoding}_loss_curve_iter.pkl")
        with open(pkl_file, 'wb') as pck:
            pickle.dump(loss_history, pck)

        fig_file = os.path.join(self.result_folder, f"{self.drug_encoding}_{self.target_encoding}_loss_curve.png")
        plt.savefig(fig_file)
        if verbose:
            print('--- Training Finished ---')

    def predict(self, df_data):
        print('predicting...')
        dataset = data_process_loader(df_data.index.values, df_data.Label.values, df_data, **self.config)
        dataset = dataset.batch(self.config['batch_size'], drop_remainder=False)
        score = self.test(dataset.make_one_shot_iterator(), repurposing_mode=True)
        return score
    
    def load_pretrained(self, path_dir):
        weight_file = os.path.join(path_dir, 'model.h5')
        self.model.load_weights(weight_file)
        self.binary = self.config['binary']

    def plot_roc_auc(self, y_pred, y_label):
        roc_auc_file = os.path.join(self.result_folder, "roc-auc.jpg")
        plt.figure(0)
        roc_curve(y_pred, y_label, roc_auc_file, self.drug_encoding + '_' + self.target_encoding)

    def plot_pr_auc(self, y_pred, y_label):
        plt.figure(1)
        pr_auc_file = os.path.join(self.result_folder, "pr-auc.jpg")
        prauc_curve(y_pred, y_label, pr_auc_file, self.drug_encoding + '_' + self.target_encoding)