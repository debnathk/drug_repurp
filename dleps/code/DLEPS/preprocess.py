import pandas as pd

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

# Save dataset
df_want.to_csv("../../data/bindingDB_IC50.csv", index=False)