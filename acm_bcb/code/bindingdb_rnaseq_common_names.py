from utils import *
import pandas as pd

df_drug_names = pd.read_csv('../../../data/gene_expression_drug_names.csv')
drug_names = list(df_drug_names['DRUG_NAME'])
print(drug_names)

df_bindingdb = pd.read_csv('../data/BindingDB_IC50_human_subset.csv')

# Unique smiles in bindingdb dataset

unique_smiles = set(df_bindingdb['Ligand SMILES'])
print(len(unique_smiles))

bindingdb_drug_names = []
for smiles in unique_smiles:
    try:
        names = smiles2iupac(smiles)
        bindingdb_drug_names.append(names)
    except:
        print("SMILES not found error")

print(bindingdb_drug_names)

