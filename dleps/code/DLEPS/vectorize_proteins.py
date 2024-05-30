import pandas as pd
import numpy as np
import utils
import h5py

# Load dataset
df = pd.read_csv('../../data/bindingDB_IC50.csv')

## Extract drug SMILES, protein sequences and IC50 data
df_proteins = df['proteins']
print(len(df_proteins))
# One-hot encoding of protein sequences

AA = pd.Series(df_proteins.unique()).apply(utils.protein2onehot)
AA_dict = dict(zip(df_proteins.unique(), AA))
df_proteins = [AA_dict[i] for i in df_proteins]
one_hot_proteins = np.array(df_proteins)
print(one_hot_proteins.shape)

h5f = h5py.File('../../data/proteins.h5', 'w')
h5f.create_dataset('data', data=one_hot_proteins)
h5f.close()