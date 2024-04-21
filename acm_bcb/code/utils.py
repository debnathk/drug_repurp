#### Utility function definitions ####

from rdkit.Chem import MolToSmiles, MolFromSmiles
import numpy as np
import requests
import pandas as pd
import h5py

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
import torch
import torch.nn.functional as F
from torch.utils import data


def name2cid(compound_name):
    """
    Retrieves the SMILES string for a given compound name using the PubChem REST API.

    Args:
        compound_name (str): The name of the compound.

    Returns:
        str: The SMILES string for the specified compound, or None if not found.
    """
    try:
        # Construct the URL for the PubChem REST API
        url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{compound_name}/cids/TXT"


        # Send a GET request to the API
        response = requests.get(url)

        # Check if the request was successful
        if response.status_code == 200:
            smiles = response.text.strip()
            return smiles
        else:
            print(f"Error retrieving SMILES for '{compound_name}': {response.status_code} - {response.text}")
            return None
    except Exception as e:
        print(f"Error retrieving SMILES for '{compound_name}': {e}")
        return None
	
def cid2smiles(cid):
    """
    Retrieves the SMILES string for a given compound name using the PubChem REST API.

    Args:
        compound_name (str): The name of the compound.

    Returns:
        str: The SMILES string for the specified compound, or None if not found.
    """
    try:
        # Construct the URL for the PubChem REST API
        url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid}/property/CanonicalSMILES/TXT"


        # Send a GET request to the API
        response = requests.get(url)

        # Check if the request was successful
        if response.status_code == 200:
            smiles = response.text.strip()
            return smiles
        else:
            print(f"Error retrieving SMILES for '{cid}': {response.status_code} - {response.text}")
            return None
    except Exception as e:
        print(f"Error retrieving SMILES for '{cid}': {e}")
        return None


def standardize_smiles(smiles):
    try:
        if smiles is not np.nan:
            return MolToSmiles(MolFromSmiles(smiles))
    except:
        return smiles

def smiles2pertid(df, smiles_col, pertid_col):
    pert_ids = []
    SMILES = df[smiles_col]
    for smiles in SMILES:
        pert_id = (df.loc[df[smiles_col] == smiles, pertid_col]).values[0]
        pert_ids.append(pert_id)
    return pert_ids

def smiles2cmap(df, smiles_col, cmap_col):
    cmaps = []
    SMILES = df[smiles_col]
    for smiles in SMILES:
        cmap = (df.loc[df[smiles_col] == smiles, cmap_col]).values[0]
        cmaps.append(cmap)
    return cmaps

def smiles2alias(df, smiles_col, alias_col):
    aliases = []
    SMILES = df[smiles_col]
    for smiles in SMILES:
        alias = (df.loc[df[smiles_col] == smiles, alias_col]).values[0]
        aliases.append(alias)
    return aliases

def check_substring(df_query, query_col_name, substrings):
	filtered_df = df_query[df_query[query_col_name].apply(lambda x: any(substring in str(x).lower() for substring in [s.lower() for s in substrings]))]
	return filtered_df

def drug_2_embed(x):
	return enc_drug.transform(np.array(x).reshape(-1,1)).toarray().T 

def protein_2_embed(x):
	return enc_protein.transform(np.array(x).reshape(-1,1)).toarray().T

class data_process_loader(data.Dataset):

	def __init__(self, list_IDs, labels, df, **config):
		'Initialization'
		self.labels = labels
		self.list_IDs = list_IDs
		self.df = df
		self.config = config

		if self.config['drug_encoding'] in ['DGL_GCN', 'DGL_NeuralFP']:
			from dgllife.utils import smiles_to_bigraph, CanonicalAtomFeaturizer, CanonicalBondFeaturizer
			self.node_featurizer = CanonicalAtomFeaturizer()
			self.edge_featurizer = CanonicalBondFeaturizer(self_loop = True)
			from functools import partial
			self.fc = partial(smiles_to_bigraph, add_self_loop=True)

		elif self.config['drug_encoding'] == 'DGL_AttentiveFP':
			from dgllife.utils import smiles_to_bigraph, AttentiveFPAtomFeaturizer, AttentiveFPBondFeaturizer
			self.node_featurizer = AttentiveFPAtomFeaturizer()
			self.edge_featurizer = AttentiveFPBondFeaturizer(self_loop=True)
			from functools import partial
			self.fc = partial(smiles_to_bigraph, add_self_loop=True)

		elif self.config['drug_encoding'] in ['DGL_GIN_AttrMasking', 'DGL_GIN_ContextPred']:
			from dgllife.utils import smiles_to_bigraph, PretrainAtomFeaturizer, PretrainBondFeaturizer
			self.node_featurizer = PretrainAtomFeaturizer()
			self.edge_featurizer = PretrainBondFeaturizer()
			from functools import partial
			self.fc = partial(smiles_to_bigraph, add_self_loop=True)

	def __len__(self):
		'Denotes the total number of samples'
		return len(self.list_IDs)

	def __getitem__(self, index):
		'Generates one sample of data'
		index = self.list_IDs[index]
		v_d = self.df.iloc[index]['drug_encoding']        
		if self.config['drug_encoding'] == 'CNN' or self.config['drug_encoding'] == 'CNN_RNN':
			v_d = drug_2_embed(v_d)
		elif self.config['drug_encoding'] in ['DGL_GCN', 'DGL_NeuralFP', 'DGL_GIN_AttrMasking', 'DGL_GIN_ContextPred', 'DGL_AttentiveFP']:
			v_d = self.fc(smiles = v_d, node_featurizer = self.node_featurizer, edge_featurizer = self.edge_featurizer)
		v_p = self.df.iloc[index]['target_encoding']
		if self.config['target_encoding'] == 'CNN' or self.config['target_encoding'] == 'CNN_RNN':
			v_p = protein_2_embed(v_p)
		y = self.labels[index]
		return v_d, v_p, y


def generate_config(drug_encoding = None, target_encoding = None, 
					result_folder = "./result/",
					input_dim_drug = 1024, 
					input_dim_protein = 8420,
					hidden_dim_drug = 256, 
					hidden_dim_protein = 256,
					cls_hidden_dims = [1024, 1024, 512],
					mlp_hidden_dims_drug = [1024, 256, 64],
					mlp_hidden_dims_target = [1024, 256, 64],
					batch_size = 256,
					train_epoch = 10,
					test_every_X_epoch = 20,
					LR = 1e-4,
					decay = 0,
					transformer_emb_size_drug = 128,
					transformer_intermediate_size_drug = 512,
					transformer_num_attention_heads_drug = 8,
					transformer_n_layer_drug = 8,
					transformer_emb_size_target = 64,
					transformer_intermediate_size_target = 256,
					transformer_num_attention_heads_target = 4,
					transformer_n_layer_target = 2,
					transformer_dropout_rate = 0.1,
					transformer_attention_probs_dropout = 0.1,
					transformer_hidden_dropout_rate = 0.1,
					mpnn_hidden_size = 50,
					mpnn_depth = 3,
					cnn_drug_filters = [32,64,96],
					cnn_drug_kernels = [4,6,8],
					cnn_target_filters = [32,64,96],
					cnn_target_kernels = [4,8,12],
					rnn_Use_GRU_LSTM_drug = 'GRU',
					rnn_drug_hid_dim = 64,
					rnn_drug_n_layers = 2,
					rnn_drug_bidirectional = True,
					rnn_Use_GRU_LSTM_target = 'GRU',
					rnn_target_hid_dim = 64,
					rnn_target_n_layers = 2,
					rnn_target_bidirectional = True,
					num_workers = 0,
					cuda_id = None,
					gnn_hid_dim_drug = 64,
					gnn_num_layers = 3,
					gnn_activation = F.relu,
					neuralfp_max_degree = 10,
					neuralfp_predictor_hid_dim = 128,
					neuralfp_predictor_activation = torch.tanh,
					attentivefp_num_timesteps = 2
					):

	base_config = {'input_dim_drug': input_dim_drug,
					'input_dim_protein': input_dim_protein,
					'hidden_dim_drug': hidden_dim_drug, # hidden dim of drug
					'hidden_dim_protein': hidden_dim_protein, # hidden dim of protein
					'cls_hidden_dims' : cls_hidden_dims, # decoder classifier dim 1
					'batch_size': batch_size,
					'train_epoch': train_epoch,
					'test_every_X_epoch': test_every_X_epoch, 
					'LR': LR,
					'drug_encoding': drug_encoding,
					'target_encoding': target_encoding, 
					'result_folder': result_folder,
					'binary': False,
					'num_workers': num_workers,
					'cuda_id': cuda_id                 
	}
	if not os.path.exists(base_config['result_folder']):
		os.makedirs(base_config['result_folder'])

	if drug_encoding == 'CNN':
		base_config['cnn_drug_filters'] = cnn_drug_filters
		base_config['cnn_drug_kernels'] = cnn_drug_kernels

	elif drug_encoding is None:
		pass
	else:
		raise AttributeError("Please use the correct drug encoding available!")
	# if target_encoding == 'AAC':
	# 	base_config['mlp_hidden_dims_target'] = mlp_hidden_dims_target # MLP classifier dim 1				
	# elif target_encoding == 'PseudoAAC':
	# 	base_config['input_dim_protein'] = 30
	# 	base_config['mlp_hidden_dims_target'] = mlp_hidden_dims_target # MLP classifier dim 1				
	# elif target_encoding == 'Conjoint_triad':
	# 	base_config['input_dim_protein'] = 343
	# 	base_config['mlp_hidden_dims_target'] = mlp_hidden_dims_target # MLP classifier dim 1				
	# elif target_encoding == 'Quasi-seq':
	# 	base_config['input_dim_protein'] = 100
	# 	base_config['mlp_hidden_dims_target'] = mlp_hidden_dims_target # MLP classifier dim 1	
	# elif target_encoding == 'ESPF':
	# 	base_config['input_dim_protein'] = len(idx2word_p)
	# 	base_config['mlp_hidden_dims_target'] = mlp_hidden_dims_target # MLP classifier dim 1			
	if target_encoding == 'CNN':
		base_config['cnn_target_filters'] = cnn_target_filters
		base_config['cnn_target_kernels'] = cnn_target_kernels
	# elif target_encoding == 'CNN_RNN':
	# 	base_config['rnn_Use_GRU_LSTM_target'] = rnn_Use_GRU_LSTM_target
	# 	base_config['rnn_target_hid_dim'] = rnn_target_hid_dim
	# 	base_config['rnn_target_n_layers'] = rnn_target_n_layers
	# 	base_config['rnn_target_bidirectional'] = rnn_target_bidirectional 
	# 	base_config['cnn_target_filters'] = cnn_target_filters
	# 	base_config['cnn_target_kernels'] = cnn_target_kernels
	# elif target_encoding == 'Transformer':
	# 	base_config['input_dim_protein'] = 4114
	# 	base_config['transformer_emb_size_target'] = transformer_emb_size_target
	# 	base_config['transformer_num_attention_heads_target'] = transformer_num_attention_heads_target
	# 	base_config['transformer_intermediate_size_target'] = transformer_intermediate_size_target
	# 	base_config['transformer_n_layer_target'] = transformer_n_layer_target	
	# 	base_config['transformer_dropout_rate'] = transformer_dropout_rate
	# 	base_config['transformer_attention_probs_dropout'] = transformer_attention_probs_dropout
	# 	base_config['transformer_hidden_dropout_rate'] = transformer_hidden_dropout_rate
	# 	base_config['hidden_dim_protein'] = transformer_emb_size_target
	elif target_encoding is None:
		pass
	else:
		raise AttributeError("Please use the correct protein encoding available!")

	return base_config

def convert_y_unit(y, from_, to_):
	array_flag = False
	if isinstance(y, (int, float)):
		y = np.array([y])
		array_flag = True
	y = y.astype(float)    
	# basis as nM
	if from_ == 'nM':
		y = y
	elif from_ == 'p':
		y = 10**(-y) / 1e-9

	if to_ == 'p':
		zero_idxs = np.where(y == 0.)[0]
		y[zero_idxs] = 1e-10
		y = -np.log10(y*1e-9)
	elif to_ == 'nM':
		y = y
        
	if array_flag:
		return y[0]
	return y

# '?' padding
amino_char = ['?', 'A', 'C', 'B', 'E', 'D', 'G', 'F', 'I', 'H', 'K', 'M', 'L', 'O',
       'N', 'Q', 'P', 'S', 'R', 'U', 'T', 'W', 'V', 'Y', 'X', 'Z']

smiles_char = ['?', '#', '%', ')', '(', '+', '-', '.', '1', '0', '3', '2', '5', '4',
       '7', '6', '9', '8', '=', 'A', 'C', 'B', 'E', 'D', 'G', 'F', 'I',
       'H', 'K', 'M', 'L', 'O', 'N', 'P', 'S', 'R', 'U', 'T', 'W', 'V',
       'Y', '[', 'Z', ']', '_', 'a', 'c', 'b', 'e', 'd', 'g', 'f', 'i',
       'h', 'm', 'l', 'o', 'n', 's', 'r', 'u', 't', 'y']

from sklearn.preprocessing import OneHotEncoder
enc_protein = OneHotEncoder().fit(np.array(amino_char).reshape(-1, 1))
enc_drug = OneHotEncoder().fit(np.array(smiles_char).reshape(-1, 1))

MAX_SEQ_PROTEIN = 2000
MAX_SEQ_DRUG = 277

# random_fold
def create_fold(df, fold_seed, frac):
    train_frac, val_frac, test_frac = frac
    test = df.sample(frac = test_frac, replace = False, random_state = fold_seed)
    train_val = df[~df.index.isin(test.index)]
    val = train_val.sample(frac = val_frac/(1-test_frac), replace = False, random_state = 1)
    train = train_val[~train_val.index.isin(val.index)]
    
    return train, val, test

# cold protein
def create_fold_setting_cold_protein(df, fold_seed, frac):
    train_frac, val_frac, test_frac = frac
    gene_drop = df['Target Sequence'].drop_duplicates().sample(frac = test_frac, replace = False, random_state = fold_seed).values
    
    test = df[df['Target Sequence'].isin(gene_drop)]

    train_val = df[~df['Target Sequence'].isin(gene_drop)]
    
    gene_drop_val = train_val['Target Sequence'].drop_duplicates().sample(frac = val_frac/(1-test_frac), 
    																	  replace = False, 
    																	  random_state = fold_seed).values
    val = train_val[train_val['Target Sequence'].isin(gene_drop_val)]
    train = train_val[~train_val['Target Sequence'].isin(gene_drop_val)]
    
    return train, val, test

# cold drug
def create_fold_setting_cold_drug(df, fold_seed, frac):
    train_frac, val_frac, test_frac = frac
    drug_drop = df['SMILES'].drop_duplicates().sample(frac = test_frac, replace = False, random_state = fold_seed).values
    
    test = df[df['SMILES'].isin(drug_drop)]

    train_val = df[~df['SMILES'].isin(drug_drop)]
    
    drug_drop_val = train_val['SMILES'].drop_duplicates().sample(frac = val_frac/(1-test_frac), 
    															 replace = False, 
    															 random_state = fold_seed).values
    val = train_val[train_val['SMILES'].isin(drug_drop_val)]
    train = train_val[~train_val['SMILES'].isin(drug_drop_val)]
    
    return train, val, test

def gVAE(smiles):
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
	# assert type(X_drugs) == list
	tokens = map(_tokenize, smiles)
	tokens_list = []
	for t in tokens:
		tokens_list.append(t[0])

	return tokens_list

def trans_drug(x):
	temp = list(x)
	temp = [i if i in smiles_char else '?' for i in temp]
	if len(temp) < MAX_SEQ_DRUG:
		temp = temp + ['?'] * (MAX_SEQ_DRUG-len(temp))
	else:
		temp = temp [:MAX_SEQ_DRUG]
	return temp

def trans_protein(x):
	temp = list(x.upper())
	temp = [i if i in amino_char else '?' for i in temp]
	if len(temp) < MAX_SEQ_PROTEIN:
		temp = temp + ['?'] * (MAX_SEQ_PROTEIN-len(temp))
	else:
		temp = temp [:MAX_SEQ_PROTEIN]
	return temp

def encode_drug(df_data, drug_encoding, column_name = 'SMILES', save_column_name = 'drug_encoding'):
	print('encoding drug...')
	print('unique drugs: ' + str(len(df_data[column_name].unique())))
	if drug_encoding == 'gVAE':
		unique = pd.Series(df_data[column_name].unique()).apply(gVAE)
		unique_dict = dict(zip(df_data[column_name].unique(), unique))
		df_data[save_column_name] = [unique_dict[i] for i in df_data[column_name]]
	elif drug_encoding == 'CNN':
		unique = pd.Series(df_data[column_name].unique()).apply(trans_drug)
		unique_dict = dict(zip(df_data[column_name].unique(), unique))
		df_data[save_column_name] = [unique_dict[i] for i in df_data[column_name]]
	else:
		raise AttributeError("Please use the correct drug encoding available!")
	return df_data

def encode_protein(df_data, target_encoding, column_name = 'Target Sequence', save_column_name = 'target_encoding'):
	print('encoding protein...')
	print('unique target sequence: ' + str(len(df_data[column_name].unique())))
	if target_encoding == 'CNN':
		AA = pd.Series(df_data[column_name].unique()).apply(trans_protein)
		AA_dict = dict(zip(df_data[column_name].unique(), AA))
		df_data[save_column_name] = [AA_dict[i] for i in df_data[column_name]]
		# the embedding is large and not scalable but quick, so we move to encode in dataloader batch. 
	else:
		raise AttributeError("Please use the correct protein encoding available!")
	return df_data

def data_process(X_drug = None, X_target = None, y=None, drug_encoding=None, target_encoding=None, 
				 split_method = 'random', frac = [0.7, 0.1, 0.2], random_seed = 1, sample_frac = 1, mode = 'DTI', X_drug_ = None, X_target_ = None):
	
	if random_seed == 'TDC':
		random_seed = 1234
	#property_prediction_flag = X_target is None
	# property_prediction_flag, function_prediction_flag, DDI_flag, PPI_flag, DTI_flag = False, False, False, False, False
	DTI_flag = False

	if (X_drug is not None) and (X_target is not None):
		DTI_flag = True
		if (X_drug is None) or (X_target is None):
			raise AttributeError("Target pair sequence should be in X_target, X_drug")
	else:
		raise AttributeError("Please use the correct mode - DTI")

	# if split_method == 'repurposing_VS':
	# 	y = [-1]*len(X_drug) # create temp y for compatitibility
	
	if DTI_flag:
		print('Drug Target Interaction Prediction Mode...')
		if isinstance(X_target, str):
			X_target = [X_target]
		# if len(X_target) == 1:
		# 	# one target high throughput screening setting
		# 	X_target = np.tile(X_target, (length_func(X_drug), ))

		df_data = pd.DataFrame(zip(X_drug, X_target, y))
		df_data.rename(columns={0:'SMILES',
								1: 'Target Sequence',
								2: 'Label'}, 
								inplace=True)
		print('in total: ' + str(len(df_data)) + ' drug-target pairs')


	if sample_frac != 1:
		df_data = df_data.sample(frac = sample_frac).reset_index(drop = True)
		print('after subsample: ' + str(len(df_data)) + ' data points...') 

	if DTI_flag:
		df_data = encode_drug(df_data, drug_encoding)
		df_data = encode_protein(df_data, target_encoding)

	# dti split
	if DTI_flag:
		if split_method == 'repurposing_VS':
			pass
		else:
			print('splitting dataset...')

		if split_method == 'random': 
			train, val, test = create_fold(df_data, random_seed, frac)
		elif split_method == 'cold_drug':
			train, val, test = create_fold_setting_cold_drug(df_data, random_seed, frac)
		elif split_method == 'HTS':
			train, val, test = create_fold_setting_cold_drug(df_data, random_seed, frac)
			val = pd.concat([val[val.Label == 1].drop_duplicates(subset = 'SMILES'), val[val.Label == 0]])
			test = pd.concat([test[test.Label == 1].drop_duplicates(subset = 'SMILES'), test[test.Label == 0]])        
		elif split_method == 'cold_protein':
			train, val, test = create_fold_setting_cold_protein(df_data, random_seed, frac)
		elif split_method == 'repurposing_VS':
			train = df_data
			val = df_data
			test = df_data
		elif split_method == 'no_split':
			print('do not do train/test split on the data for already splitted data')
			return df_data.reset_index(drop=True)
		else:
			raise AttributeError("Please select one of the three split method: random, cold_drug, cold_target!")

	print('Done.')
	return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)