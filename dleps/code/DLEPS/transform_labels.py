import pandas as pd
import numpy as np
import utils
import h5py

# Load dataset
df = pd.read_csv('../../data/bindingDB_IC50.csv')

## Extract drug SMILES, protein sequences and IC50 data
df_labels = df['labels']
print(len(df_labels))

# Log-transform labels
L_dict = pd.Series(df_labels)
l_converted = L_dict.apply(lambda l: utils.convert_y_unit(l, from_='nM', to_='p')).tolist()
np.shape(l_converted)

# Save dataset
h5f = h5py.File('../../data/pIC50.h5', 'w')
h5f.create_dataset('data', data=l_converted)
h5f.close()