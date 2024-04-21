# Encoding of target proteins is adopted form DeepPurpose - https://github.com/kexinhuang12345/DeepPurpose/tree/master
import pandas as pd
import torch
from torch.autograd import Variable
import torch.nn.functional as F
from torch.utils import data
from torch.utils.data import SequentialSampler
from torch import nn 

# '?' padding
amino_char = ['?', 'A', 'C', 'B', 'E', 'D', 'G', 'F', 'I', 'H', 'K', 'M', 'L', 'O',
       'N', 'Q', 'P', 'S', 'R', 'U', 'T', 'W', 'V', 'Y', 'X', 'Z']

MAX_SEQ_PROTEIN = 1000

def trans_protein(x):
	temp = list(x.upper())
	temp = [i if i in amino_char else '?' for i in temp]
	if len(temp) < MAX_SEQ_PROTEIN:
		temp = temp + ['?'] * (MAX_SEQ_PROTEIN-len(temp))
	else:
		temp = temp [:MAX_SEQ_PROTEIN]
	return temp

def encode_protein(df_data, target_encoding='CNN', column_name = 'BindingDB Target Chain Sequence', save_column_name = 'target_encoding'):
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

class CNN(nn.Sequential):
	def __init__(self, encoding, **config):
		super(CNN, self).__init__()
		if encoding == 'drug':
			in_ch = [63] + config['cnn_drug_filters']
			kernels = config['cnn_drug_kernels']
			layer_size = len(config['cnn_drug_filters'])
			self.conv = nn.ModuleList([nn.Conv1d(in_channels = in_ch[i], 
													out_channels = in_ch[i+1], 
													kernel_size = kernels[i]) for i in range(layer_size)])
			self.conv = self.conv.double()
			n_size_d = self._get_conv_output((63, 100))
			#n_size_d = 1000
			self.fc1 = nn.Linear(n_size_d, config['hidden_dim_drug'])

		if encoding == 'protein':
			in_ch = [26] + config['cnn_target_filters']
			kernels = config['cnn_target_kernels']
			layer_size = len(config['cnn_target_filters'])
			self.conv = nn.ModuleList([nn.Conv1d(in_channels = in_ch[i], 
													out_channels = in_ch[i+1], 
													kernel_size = kernels[i]) for i in range(layer_size)])
			self.conv = self.conv.double()
			n_size_p = self._get_conv_output((26, 1000))

			self.fc1 = nn.Linear(n_size_p, config['hidden_dim_protein'])

	def _get_conv_output(self, shape):
		bs = 1
		input = Variable(torch.rand(bs, *shape))
		output_feat = self._forward_features(input.double())
		n_size = output_feat.data.view(bs, -1).size(1)
		return n_size

	def _forward_features(self, x):
		for l in self.conv:
			x = F.relu(l(x))
		x = F.adaptive_max_pool1d(x, output_size=1)
		return x

	def forward(self, v):
		v = self._forward_features(v.double())
		v = v.view(v.size(0), -1)
		v = self.fc1(v.float())
		return v

if __name__ == "__main__":
	df = pd.read_csv('../data/BindingDB_IC50_human.csv')
	df_sample = df.head(10)
	df_sample_encoded = encode_protein(df_sample)
	df_sample_encoded.to_csv('../data/sample_encoding.csv', index=False)