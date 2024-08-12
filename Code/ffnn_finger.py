import torch
from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np
import pandas as pd
from dgllife.model import model_zoo
import torch.utils.data as data
from dgllife.utils import smiles_to_bigraph
from dgllife.utils import EarlyStopping, Meter
from dgllife.utils import AttentiveFPAtomFeaturizer
from dgllife.utils import AttentiveFPBondFeaturizer
from dgllife.data import MoleculeCSVDataset
import pandas as pd
import torch.nn as nn
import os


# def add_cuda_to_path():
#     if os.name != "nt":
#         return
#     path = os.getenv("PATH")
#     if not path:
#         return
#     path_split = path.split(";")
#     for folder in path_split:
#         if "cuda" in folder.lower() or "tensorrt" in folder.lower():
#             os.add_dll_directory(folder)
def add_cuda_to_path():
    if os.name != "nt":
        return
    path = os.getenv("PATH")
    if not path:
        return
    cuda_folders = []
    path_split = path.split(";")
    for folder in path_split:
        if "cuda" in folder.lower() or "tensorrt" in folder.lower():
            cuda_folders.append(folder)
   
    if cuda_folders:
        for cuda_folder in cuda_folders:
            
            if hasattr(os, 'add_dll_directory'):
                os.add_dll_directory(cuda_folder)
            else:
                
                os.environ["PATH"] += os.pathsep + cuda_folder


add_cuda_to_path()



def finger(path):
    finger = pd.read_excel(path).iloc[:,:-1].values
    """finger = []
    with open(path, 'r', encoding='utf-8') as f:
        for ann in f.readlines():
            ann = ann.strip('\n') 
            # print(ann)
            mol = Chem.MolFromSmiles(ann)
            fp = AllChem.GetMorganFingerprint(mol, 2)
           
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=108)
            # fp = fp.ToBitString()
            finger.append(np.array(fp))
    finger = np.array(finger)
    finger = finger.astype(np.float32)
    """
    return finger



def trans(path):
    data = np.load(path)
    data = data.astype(np.float32)
    return data


class Net(torch.nn.Module):  
    def __init__(self):  
        super(Net, self).__init__()  
        self.atom_featurizer = AttentiveFPAtomFeaturizer(atom_data_field='hv')
        self.bond_featurizer = AttentiveFPBondFeaturizer(bond_data_field='he')
        self.n_feats = self.atom_featurizer.feat_size('hv')
        self.e_feats = self.bond_featurizer.feat_size('he')
       
        self.model = model_zoo.AttentiveFPPredictor(node_feat_size=self.n_feats,
                                                    edge_feat_size=self.e_feats,
                                                    num_layers=2,
                                                    num_timesteps=1,
                                                    graph_feat_size=300,
                                                    n_tasks=4,
                                                    dropout=0.3
                                                    )
       
        self.layer = nn.Linear(300, 108, bias=True)
        self.lstm = nn.LSTM(input_size=108, hidden_size=108, batch_first=True)
        self.ffnn = nn.Linear(in_features=108, out_features=108)
        _set_module(self.model, 'predict.1', self.layer)
      
        self.avgpool = nn.AvgPool1d(3, 2)
   
        self.fc1 = nn.Linear(363, 1024, bias=True)
        self.fc2 = nn.Linear(1024, 512, bias=True)
        self.fc3 = nn.Linear(512, 256, bias=True)
        self.fc4 = nn.Linear(256, 1, bias=True)
       
        self.bn1 = nn.BatchNorm1d(1024)
        self.bn2 = nn.BatchNorm1d(512)
        self.bn3 = nn.BatchNorm1d(256)
    
        self.rl = nn.ReLU()

    def forward(self, bg, n_feats, e_feats, trans, finger):
        
        x = self.model(bg, n_feats, e_feats)
        # print(x.dtype,trans.dtype,finger.dtype)
       

        finger = finger.float()
        finger = self.ffnn(finger)  # Apply FFNN layer (ignoring outputted hidden state)

        #finger = finger.view(-1, 1, 108).float()  # Reshape the tensor to fit the LSTM layer
        #finger, _ = self.lstm(finger)  # Apply LSTM layer (ignoring outputted hidden state)
        #finger = finger.view(-1, 108)  # Get the tensor back to its original shape before cats

        x1 = torch.cat((x, trans, finger), 1)
        x = self.avgpool(x1)
       
        x = self.bn1(self.fc1(x))
        x = self.rl(x)

        x = self.bn2(self.fc2(x))
        x = self.rl(x)

        x = self.bn3(self.fc3(x))
        x = self.rl(x)

        x = self.fc4(x)

        return x, x1


def _set_module(model, submodule_key, module):
    tokens = submodule_key.split('.')
    sub_tokens = tokens[:-1]
    cur_mod = model
    for s in sub_tokens:
        cur_mod = getattr(cur_mod, s)
    setattr(cur_mod, tokens[-1], module)
