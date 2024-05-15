from rdkit.Chem import MolToSmiles, MolFromSmiles
import numpy as np

def standardize(smiles):
    try:
        if smiles is not np.nan:
            return MolToSmiles(MolFromSmiles(smiles))
    except:
        return None
 

