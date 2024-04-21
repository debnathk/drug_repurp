import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from rdkit.Chem.Fingerprints import FingerprintMols
from rdkit.Chem.rdReducedGraphs import GetErGFingerprint
from DeepPurpose.pybiomed_helper import _GetPseudoAAC, CalculateAADipeptideComposition, \
calcPubChemFingerAll, CalculateConjointTriad, GetQuasiSequenceOrder
import torch
from torch.utils import data
from torch.autograd import Variable
import torch.nn.functional as F

try:
	from descriptastorus.descriptors import rdDescriptors, rdNormalizedDescriptors
except:
	raise ImportError("Please install pip install git+https://github.com/bp-kelley/descriptastorus and pip install pandas-flavor")
from DeepPurpose.chemutils import get_mol, atom_features, bond_features, MAX_NB, ATOM_FDIM, BOND_FDIM
from subword_nmt.apply_bpe import BPE
import codecs
import pickle
import wget
from zipfile import ZipFile 
import os
import sys
import pathlib

this_dir = str(pathlib.Path(__file__).parent.absolute())

MAX_ATOM = 600
MAX_BOND = MAX_ATOM * 2

# ESPF encoding
vocab_path = f"{this_dir}/ESPF/drug_codes_chembl_freq_1500.txt"
bpe_codes_drug = codecs.open(vocab_path)
dbpe = BPE(bpe_codes_drug, merges=-1, separator='')
sub_csv = pd.read_csv(f"{this_dir}/ESPF/subword_units_map_chembl_freq_1500.csv")

idx2word_d = sub_csv['index'].values
words2idx_d = dict(zip(idx2word_d, range(0, len(idx2word_d))))

vocab_path = f"{this_dir}/ESPF/protein_codes_uniprot_2000.txt"
bpe_codes_protein = codecs.open(vocab_path)
pbpe = BPE(bpe_codes_protein, merges=-1, separator='')
#sub_csv = pd.read_csv(dataFolder + '/subword_units_map_protein.csv')
sub_csv = pd.read_csv(f"{this_dir}/ESPF/subword_units_map_uniprot_2000.csv")

idx2word_p = sub_csv['index'].values
words2idx_p = dict(zip(idx2word_p, range(0, len(idx2word_p))))

from DeepPurpose.chemutils import get_mol, atom_features, bond_features, MAX_NB

def encode_protein(df_data, target_encoding, column_name = 'Target Sequence', save_column_name = 'target_encoding'):
	print('encoding protein...')
	print('unique target sequence: ' + str(len(df_data[column_name].unique())))
	if target_encoding == 'AAC':
		print('-- Encoding AAC takes time. Time Reference: 24s for ~100 sequences in a CPU.\
				 Calculate your time by the unique target sequence #, instead of the entire dataset.')
		AA = pd.Series(df_data[column_name].unique()).apply(target2aac)
		AA_dict = dict(zip(df_data[column_name].unique(), AA))
		df_data[save_column_name] = [AA_dict[i] for i in df_data[column_name]]
	elif target_encoding == 'PseudoAAC':
		print('-- Encoding PseudoAAC takes time. Time Reference: 462s for ~100 sequences in a CPU.\
				 Calculate your time by the unique target sequence #, instead of the entire dataset.')
		AA = pd.Series(df_data[column_name].unique()).apply(target2paac)
		AA_dict = dict(zip(df_data[column_name].unique(), AA))
		df_data[save_column_name] = [AA_dict[i] for i in df_data[column_name]]
	elif target_encoding == 'Conjoint_triad':
		AA = pd.Series(df_data[column_name].unique()).apply(target2ct)
		AA_dict = dict(zip(df_data[column_name].unique(), AA))
		df_data[save_column_name] = [AA_dict[i] for i in df_data[column_name]]
	elif target_encoding == 'Quasi-seq':
		AA = pd.Series(df_data[column_name].unique()).apply(target2quasi)
		AA_dict = dict(zip(df_data[column_name].unique(), AA))
		df_data[save_column_name] = [AA_dict[i] for i in df_data[column_name]]
	elif target_encoding == 'ESPF':
		AA = pd.Series(df_data[column_name].unique()).apply(protein2espf)
		AA_dict = dict(zip(df_data[column_name].unique(), AA))
		df_data[save_column_name] = [AA_dict[i] for i in df_data[column_name]]
	elif target_encoding == 'CNN':
		AA = pd.Series(df_data[column_name].unique()).apply(trans_protein)
		AA_dict = dict(zip(df_data[column_name].unique(), AA))
		df_data[save_column_name] = [AA_dict[i] for i in df_data[column_name]]
		# the embedding is large and not scalable but quick, so we move to encode in dataloader batch. 
	elif target_encoding == 'CNN_RNN':
		AA = pd.Series(df_data[column_name].unique()).apply(trans_protein)
		AA_dict = dict(zip(df_data[column_name].unique(), AA))
		df_data[save_column_name] = [AA_dict[i] for i in df_data[column_name]]
	elif target_encoding == 'Transformer':
		AA = pd.Series(df_data[column_name].unique()).apply(protein2emb_encoder)
		AA_dict = dict(zip(df_data[column_name].unique(), AA))
		df_data[save_column_name] = [AA_dict[i] for i in df_data[column_name]]
	else:
		raise AttributeError("Please use the correct protein encoding available!")
	return df_data