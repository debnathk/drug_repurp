from DeepPurpose import utils_tensorflow
from DeepPurpose import DTI_tensorflow as models
import warnings
import pandas as pd
warnings.filterwarnings("ignore")
df_bindingdb = pd.read_csv('../data/BindingDB_IC50_updated.csv')

selection = ['cmap_name', 'std_smiles', 'Target Name', 'BindingDB Target Chain Sequence', 'IC50 (nM)']
df_bindingdb_selection = df_bindingdb[selection]
df_bindingdb_selection['pIC50'] = df_bindingdb_selection['IC50 (nM)'].apply(lambda x: utils_tensorflow.convert_y_unit(x, 'nM', 'p'))
df_bindingdb_selection.dropna(inplace=True)
print(f'No of instances in the dataset: {len(df_bindingdb_selection)}')

X_drugs = df_bindingdb_selection['std_smiles']
X_targets = df_bindingdb_selection['BindingDB Target Chain Sequence']
y = df_bindingdb_selection['pIC50']

drug_encoding, target_encoding = 'gVAE', 'CNN'
# drug_encoding, target_encoding = 'CNN', 'CNN'

train, val, test = utils_tensorflow.data_process(X_drugs, X_targets, y, 
                                drug_encoding, target_encoding, 
                                split_method='random',frac=[0.7,0.1,0.2],
                                random_seed = 1)
print(train.head(1))

config = utils_tensorflow.generate_config(drug_encoding = drug_encoding, 
                         target_encoding = target_encoding, 
                         cls_hidden_dims = [1024,1024,512], 
                         train_epoch = 100, 
                         LR = 0.001, 
                         batch_size = 256,
                        #  gvae_input_dim_drug = 277*76,
                        #  gvae_hidden_dims_drugs = [512, 256],
                        #  gvae_latent_dim_drugs = 256,
                        #  gvae_drug_filters = [32,64,96],
                        #  gvae_drug_kernels = [4,8,12],
                         cnn_target_filters = [32,64,96],
                         cnn_target_kernels = [4,8,12]
                        )

model = models.model_initialize(**config)
print(model)

model.train(train, val, test)