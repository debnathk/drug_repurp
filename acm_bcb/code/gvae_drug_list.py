import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" 
os.environ["CUDA_VISIBLE_DEVICES"] = "4"
import sys
import molecule_vae
from rdkit.Chem import MolFromSmiles, MolToSmiles
from rdkit.Chem import Draw
import zinc_grammar
import nltk
from functools import reduce
import numpy as np  
import pandas as pd
import json

df = pd.read_csv("../data/BindingDB_IC50_updated.csv")
# print(df.columns)

smiles = list(set(df['std_smiles']))
print(f"Unique ligands: {len(smiles)}")

smiles_dict = {i: smiles[i] for i in range(len(smiles))}
with open('../data/smiles_dict.txt', 'w') as file:
    file.write(json.dumps(smiles_dict))

def xlength(y):
    return reduce(lambda sum, element: sum + 1, y, 0)

def get_zinc_tokenizer(cfg):
    long_tokens = [a for a in list(cfg._lexical_index.keys()) if xlength(a) > 1] ####
    replacements = ['$','%','^'] # ,'&']
    assert xlength(long_tokens) == len(replacements) ####xzw
    for token in replacements: 
        assert token not in cfg._lexical_index ####
    
    def tokenize(smiles):
        for i, token in enumerate(long_tokens):
            smiles = smiles.replace(token, replacements[i])
        tokens = []
        for token in smiles:
            try:
                ix = replacements.index(token)
                tokens.append(long_tokens[ix])
            except:
                tokens.append(token)
        return tokens
    
    return tokenize


_tokenize = get_zinc_tokenizer(zinc_grammar.GCFG)
_parser = nltk.ChartParser(zinc_grammar.GCFG)
_productions = zinc_grammar.GCFG.productions()
_prod_map = {}
for ix, prod in enumerate(_productions):
    _prod_map[prod] = ix
MAX_LEN = 277
_n_chars = len(_productions)

""" Encode a list of smiles strings into the latent space """
assert type(smiles) == list
tokens = map(_tokenize, smiles)
parse_trees = []
i = 0
badi = []
for t in tokens:
    #while True:
    try:
        tp = next(_parser.parse(t))
        parse_trees.append(tp)
    except:
        print("Parse tree error at %d" % i)
        badi.append(i)
    i += 1
    #print(i)
productions_seq = [tree.productions() for tree in parse_trees]
indices = [np.array([_prod_map[prod] for prod in entry], dtype=int) for entry in productions_seq]
one_hot = np.zeros((len(indices), MAX_LEN, _n_chars), dtype=np.float32)
for i in range(len(indices)):
    num_productions = len(indices[i])
    if num_productions > MAX_LEN:
        print("Too large molecules, out of range")
    #print("i=  {%d} len(indices)=  {%d} num_productions = %d " % (i,len(indices),num_productions))
        one_hot[i][np.arange(MAX_LEN),indices[i][:MAX_LEN]] = 1.
    else:    
        one_hot[i][np.arange(num_productions),indices[i]] = 1.
        one_hot[i][np.arange(num_productions, MAX_LEN),-1] = 1.

print(len(df), len(one_hot))
print(len(badi))
df = df.drop(df.iloc[badi].index)
print(len(df))

import h5py

h5f = h5py.File("../data/one_hot.h5", 'w')
h5f.create_dataset('data', data=one_hot)
h5f.close()
