import nltk
import numpy as np
import zinc_grammar
import models.model_zinc
from functools import reduce

def xlength(y):
    return reduce(lambda sum, element: sum + 1, y, 0)

def get_zinc_tokenizer(cfg):
    long_tokens = [a for a in list(cfg._lexical_index.keys()) if xlength(a) > 1]
    replacements = ['$','%','^']
    assert xlength(long_tokens) == len(replacements)
    for token in replacements:
        assert token not in cfg._lexical_index
    
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

def pop_or_nothing(S):
    try:
        return S.pop()
    except:
        return 'Nothing'

def prods_to_eq(prods):
    seq = [prods[0].lhs()]
    for prod in prods:
        if str(prod.lhs()) == 'Nothing':
            break
        for ix, s in enumerate(seq):
            if s == prod.lhs():
                seq = seq[:ix] + list(prod.rhs()) + seq[ix+1:]
                break
    try:
        return ''.join(seq)
    except:
        return ''

class ZincGrammarModel(object):

    def __init__(self, weights_file, latent_rep_size=56):
        """ Load the (trained) zinc encoder/decoder, grammar model. """
        self._grammar = zinc_grammar
        self._model = models.model_zinc
        self.MAX_LEN = self._model.MAX_LEN
        self._productions = self._grammar.GCFG.productions()
        self._prod_map = {prod: ix for ix, prod in enumerate(self._productions)}
        self._parser = nltk.ChartParser(self._grammar.GCFG)
        self._tokenize = get_zinc_tokenizer(self._grammar.GCFG)
        self._n_chars = len(self._productions)
        self._lhs_map = {lhs: ix for ix, lhs in enumerate(self._grammar.lhs_list)}
        self.vae = self._model.MoleculeVAE()
        self.vae.load(self._productions, weights_file, max_length=self.MAX_LEN, latent_rep_size=latent_rep_size)

    def onehot(self, smile):
        """ Encode a single smile string into the latent space """
        one_hot = np.zeros((self.MAX_LEN, self._n_chars), dtype=np.float32)
        tokens = self._tokenize(smile)
        parse_tree = next(self._parser.parse(tokens))
        productions_seq = parse_tree.productions()
        indices = np.array([self._prod_map[prod] for prod in productions_seq], dtype=int)
        num_productions = len(indices)
        if num_productions > self.MAX_LEN:
            print("Too large molecule, out of range")
            one_hot[np.arange(self.MAX_LEN), indices[:self.MAX_LEN]] = 1.
        else:
            one_hot[np.arange(num_productions), indices] = 1.
            one_hot[np.arange(num_productions, self.MAX_LEN), -1] = 1.
        return one_hot

    def encode(self, smile):
        one_hot = self.onehot(smile)
        one_hot = np.expand_dims(one_hot, axis=0)  # Add batch dimension
        return self.vae.encoderMV.predict(one_hot)[0]

if __name__ == "__main__":
    # Usage example:
    weights_file = '../data/vae.hdf5'
    model = ZincGrammarModel(weights_file)
    smile = "CCO"
    encoded_smile = model.encode(smile)
    print(encoded_smile)
