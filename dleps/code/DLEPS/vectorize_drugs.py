import pandas as pd
import numpy as np
import utils
import h5py

# Load dataset
df = pd.read_csv('../../data/bindingDB_IC50.csv')

## Extract drug SMILES, protein sequences and IC50 data
df_drugs = df['drugs']
print(len(df_drugs))

## Vectorizing data
# One-hot encoding of drug SMILES
S = pd.Series(df_drugs.unique()).apply(utils.standardize_smiles)
S = S.apply(utils.smiles2onehot)
S_dict = dict(zip(df_drugs.unique(), S))
df_drugs = [S_dict[i] for i in df_drugs]
one_hot_drugs = np.array(df_drugs)
print(one_hot_drugs.shape)

# Save data
h5f = h5py.File('../../data/drugs.h5', 'w')
h5f.create_dataset('data', data=one_hot_drugs)
h5f.close()