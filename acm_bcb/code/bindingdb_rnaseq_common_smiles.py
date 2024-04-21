from utils import standardize_smiles, smiles2pertid, smiles2alias, smiles2cmap, check_substring
import pandas as pd

df_l1000 = pd.read_csv('../data/compoundinfo_beta.csv')
df_bindingdb = pd.read_csv('../data/BindingDB_IC50.csv')

# Unique smiles in bindingdb dataset
# unique_smiles_bdb = set(df_bindingdb['Ligand SMILES'])
# print(len(unique_smiles_bdb))

# Standardize df_bindingdb smiles
df_bindingdb['Ligand SMILES'] = df_bindingdb['Ligand SMILES'].apply(lambda x: standardize_smiles(x))

# Unique smiles in l1000 dataset
unique_smiles_l1000 = df_l1000.drop_duplicates(subset=['canonical_smiles'])
unique_smiles_l1000['std_smiles'] = unique_smiles_l1000['canonical_smiles'].apply(lambda x: standardize_smiles(x))
print(f"Unique SMILES: {len(unique_smiles_l1000['std_smiles'])}")

# Convert smiles to pert id
# unique_pertids_l1000 = smiles2pertid(unique_smiles_l1000, 'std_smiles', 'pert_id')
# print(f'Unique pert_ids: {len(unique_pertids_l1000)}')

# Convert smiles to cmap_name
# unique_cmaps_l1000 = smiles2cmap(unique_smiles_l1000, 'std_smiles', 'cmap_name')
# print(f'Unique cmap_names: {len(unique_cmaps_l1000)}')

# Convert smiles2alias
# unique_aliases_l1000 = smiles2alias(unique_smiles_l1000, 'std_smiles', 'compound_aliases')
# print(f'Unique compound_aliases: {len(unique_aliases_l1000)}')

# Calculate common compounds
mask = df_bindingdb['Ligand SMILES'].isin(unique_smiles_l1000['std_smiles'])

df_bindingdb_filtered = df_bindingdb[mask]

print(len(df_bindingdb_filtered['Ligand SMILES']))

# Merge df_bindingdb_filtered with unique_smiles_l1000
df_merged = df_bindingdb_filtered.merge(unique_smiles_l1000, left_on='Ligand SMILES', right_on='std_smiles', how='left')

print(f'No of pert_id: {len(df_merged["pert_id"])}')
print(f'No of cmap_name: {len(df_merged["cmap_name"])}')
print(f'No of compound_aliases: {len(df_merged["compound_aliases"])}')

df_l1000_cp = pd.read_csv('../data/l1000_cp_10uM_all.csv')
print(f'Original size of L1000 cp: {len(df_l1000_cp)}')
# df_first_column = df_l1000_cp.iloc[:, :1]
# df_first_column.to_csv('../data/l1000_sample_list.csv', index=False, header=None)
# print(df_l1000_cp.columns)

substrings = set(df_merged["cmap_name"])

df_l1000_cp = check_substring(df_l1000_cp, '0', substrings)
df_l1000_cp.to_csv('../data/l1000_cp_10uM_all_filtered.csv', index=False, header=None)
print(f'Updated size of L1000 cp: {len(df_l1000_cp)}')

df_merged.to_csv('../data/BindingDB_IC50_updated.csv', index=False)