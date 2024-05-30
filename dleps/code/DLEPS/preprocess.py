import pandas as pd
import numpy as np
import utils
import h5py

# Load data
df = pd.read_csv('../../data/BindingDB_All_202403.tsv', sep='\t', on_bad_lines='skip')
# print(df.head())

## Preprocess data
# Select necessary columns

df_filtered = df[df['Number of Protein Chains in Target (>1 implies a multichain complex)'] == 1.0]
df_filtered = df_filtered[df_filtered['Ligand SMILES'].notnull()]
df_filtered = df_filtered[df_filtered['IC50 (nM)'].notnull()]
df_want = df_filtered[['BindingDB Ligand Name', 'ChEMBL ID of Ligand', 'DrugBank ID of Ligand',
 'ZINC ID of Ligand', 'Ligand SMILES', 'Target Name',
 'BindingDB Target Chain Sequence', 'Link to Ligand in BindingDB',
 'Link to Target in BindingDB', 'Link to Ligand-Target Pair in BindingDB',
 'PDB ID(s) for Ligand-Target Complex',
 'Target Source Organism According to Curator or DataSource', 'IC50 (nM)']]
df_want['IC50 (nM)'] = df_want['IC50 (nM)'].str.replace('>', '')
df_want['IC50 (nM)'] = df_want['IC50 (nM)'].str.replace('<', '')
df_want['IC50 (nM)'] = df_want['IC50 (nM)'].astype(float)

# Rename column names
df_want.rename(columns={'Ligand SMILES': 'drugs'}, inplace=True)
df_want.rename(columns={'BindingDB Target Chain Sequence': 'proteins'}, inplace=True)
df_want.rename(columns={'IC50 (nM)': 'labels'}, inplace=True)

df_want.to_csv('../../data/bindingDB_IC50.csv', index=False)

## Extract drug SMILES, protein sequences and IC50 data
df_drugs = df_want['drugs']
print(len(df_drugs))
df_proteins = df_want['proteins']
print(len(df_proteins))
df_labels = df_want['labels']
print(len(df_labels))

## Vectorizing data
# One-hot encoding of drug SMILES

S = pd.Series(df_drugs.unique()).apply(utils.standardize_smiles)
S = S.apply(utils.smiles2onehot)
S_dict = dict(zip(df_drugs.unique(), S))
df_drugs = [S_dict[i] for i in df_drugs]
one_hot_drugs = np.array(df_drugs)
print(one_hot_drugs.shape)

# One-hot encoding of protein sequences

AA = pd.Series(df_proteins.unique()).apply(utils.protein2onehot)
AA_dict = dict(zip(df_proteins.unique(), AA))
df_proteins = [AA_dict[i] for i in df_proteins]
one_hot_proteins = np.array(df_proteins)
print(one_hot_proteins.shape)

# Log-transform labels

L_dict = pd.Series(df_labels)
l_converted = L_dict.apply(lambda l: utils.convert_y_unit(l, from_='nM', to_='p')).tolist()
np.shape(l_converted)

# Save datasets

h5f = h5py.File('../../data/drugs.h5', 'w')
h5f.create_dataset('data', data=one_hot_drugs)
h5f.close()

h5f = h5py.File('../../data/proteins.h5', 'w')
h5f.create_dataset('data', data=one_hot_proteins)
h5f.close()

h5f = h5py.File('../../data/pIC50.h5', 'w')
h5f.create_dataset('data', data=l_converted)
h5f.close()

# Print dataset shapes

h5f = h5py.File('../../data/drugs.h5', 'r')
data = h5f['data'][:]
h5f.close()
print(data.shape)

h5f = h5py.File('../../data/proteins.h5', 'r')
data = h5f['data'][:]
h5f.close()
print(data.shape)

h5f = h5py.File('../../data/pIC50.h5', 'r')
data = h5f['data'][:]
h5f.close()
print(data.shape)