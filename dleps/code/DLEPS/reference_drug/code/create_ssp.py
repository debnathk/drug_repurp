import numpy as np
import pandas as pd

root = 'code/DLEPS/reference_drug/'

# Read fingerprints
df_fgrps_train = pd.read_csv(root + 'fgrps_train.csv')
df_fgrps_ref = pd.read_csv(root + 'fgrps_ref.csv')

df_fgrps_train = np.array(df_fgrps_train)
df_fgrps_ref = np.array(df_fgrps_ref)

# Calculate ssp
ssp_list = []
for i in range(df_fgrps_train.shape[0]):
    ssp = np.logical_not(np.logical_xor(df_fgrps_train[i], df_fgrps_ref))
    ssp_list.append(ssp)

ssp_stack = np.stack(ssp_list).astype(int)
print(ssp_stack.shape)

# Split train, test
# TEST_SIZE = 75
# ssp_train = ssp_stack[TEST_SIZE:]
# ssp_test = ssp_stack[:TEST_SIZE]

# Save dataset
import h5py

h5f = h5py.File(root + 'ssp_data.h5', 'w')
h5f.create_dataset('data', data=ssp_stack)
h5f.close()

# h5f = h5py.File(root + 'ssp_data_train.h5', 'w')
# h5f.create_dataset('data', data=ssp_train)
# h5f.close()

# h5f = h5py.File(root + 'ssp_data_test.h5', 'w')
# h5f.create_dataset('data', data=ssp_test)
# h5f.close()

# Read datset
h5f = h5py.File(root + 'ssp_data.h5', 'r')
ssp_data = h5f['data'][:]
# h5f = h5py.File(root + 'ssp_data_train.h5', 'r')
# ssp_train = h5f['data'][:]
# h5f = h5py.File(root + 'ssp_data_test.h5', 'r')
# ssp_test = h5f['data'][:]

print(ssp_data.shape)
# print(ssp_train.shape)
# print(ssp_test.shape)
