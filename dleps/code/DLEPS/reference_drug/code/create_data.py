import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit import DataStructs
from rdkit.ML.Cluster import Butina
from rdkit_transformation import standardize
from rdkit.Chem import Draw
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.Draw import SimilarityMaps

# root = 'code/DLEPS/dleps/code/DLEPS/reference_drug/'
root = '/lustre/home/debnathk/dleps/code/DLEPS/reference_drug/training/'
data = pd.read_csv(root + 'pubchem_50k.csv', header=None)
data = data.drop(data.columns[0], axis=1)
print(f'Total no of smiles: {len(data)}')

# Standardize smiles
molecules_pubchem = []
for smiles in data[data.columns[0]]:
    try:
        molecules_pubchem.append((Chem.MolFromSmiles(standardize(smiles))))
    except:
        molecules_pubchem.append(smiles)

# Creating fingerprints for all molecules
rdkit_gen = rdFingerprintGenerator.GetRDKitFPGenerator(maxPath=7, fpSize=3072)
fgrps_pubchem = []
for mol in molecules_pubchem:
    try:
        fgrps_pubchem.append(rdkit_gen.GetFingerprint(mol))
    except:
        print("rdkit Error occured")

print('Shape of training set fingerprints:')
print(np.array(fgrps_pubchem).shape)

df_fgrps_pubchem = pd.DataFrame(np.array(fgrps_pubchem))
df_fgrps_pubchem.columns = ["FP"+str(i) for i in range(np.array(fgrps_pubchem).shape[1])]
df_fgrps_pubchem.to_csv(root + 'fgrps_pubchem.csv', index=False)

# Creating ssp profile
df_fgrps_ref = pd.read_csv(root + 'fgrps_ref.csv')

df_fgrps_pubchem = np.array(df_fgrps_pubchem)
df_fgrps_ref = np.array(df_fgrps_ref)

# Calculate ssp
ssp_list = []
for i in range(df_fgrps_pubchem.shape[0]):
    ssp = np.logical_not(np.logical_xor(df_fgrps_pubchem[i], df_fgrps_ref))
    ssp_list.append(ssp)

ssp_stack = np.stack(ssp_list).astype(int)
print(ssp_stack.shape)

# Only keep columns with high variability


# Split train, test
TEST_SIZE = 5000
ssp_train = ssp_stack[TEST_SIZE:]
ssp_test = ssp_stack[:TEST_SIZE]

# Save dataset
import h5py

h5f = h5py.File(root + 'ssp_data_train.h5', 'w')
h5f.create_dataset('data', data=ssp_train)
h5f.close()

h5f = h5py.File(root + 'ssp_data_test.h5', 'w')
h5f.create_dataset('data', data=ssp_test)
h5f.close()

# Read datset
h5f = h5py.File(root + 'ssp_data_train.h5', 'r')
ssp_train = h5f['data'][:]
h5f = h5py.File(root + 'ssp_data_test.h5', 'r')
ssp_test = h5f['data'][:]

print(ssp_train.shape)
print(ssp_test.shape)