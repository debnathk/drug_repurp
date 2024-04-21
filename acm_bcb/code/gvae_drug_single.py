import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" 
os.environ["CUDA_VISIBLE_DEVICES"] = "4"
import zinc_grammar
import nltk
from functools import reduce
import numpy as np  

def smiles2gvaefeature(s):

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

    token = map(_tokenize, s)
    tokens = []
    for t in token:
        tokens.append(t[0])
    try:
        tp = next(_parser.parse(tokens))
    except:
        print("Parse tree error at")

    productions_seq = tp.productions()
    idx = np.array([_prod_map[prod] for prod in productions_seq], dtype=int)
    one_hot = np.zeros((MAX_LEN, _n_chars), dtype=np.float32)
    num_productions = len(idx)
    if num_productions > MAX_LEN:
            print("Too large molecules, out of range")
            one_hot[np.arange(MAX_LEN),idx[:MAX_LEN]] = 1.
    else:    
            one_hot[np.arange(num_productions),idx] = 1.
            one_hot[np.arange(num_productions, MAX_LEN),-1] = 1.

    return one_hot