import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib
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
import dgl
import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset
from dgllife.model import model_zoo
from dgllife.utils import smiles_to_bigraph
from dgllife.utils import EarlyStopping, Meter
from dgllife.utils import AttentiveFPAtomFeaturizer
from dgllife.utils import AttentiveFPBondFeaturizer
# from dgllife.data import MoleculeCSVDataset
from set import MoleculeCSVDataset
from ffnn_finger import _set_module,Net

torch.set_printoptions(threshold=np.inf)
if torch.cuda.is_available():
    print('use GPU')
    device = 'cuda'
else:
    print('use CPU')
    device = 'cpu'


import os
import random
import numpy as np
seed = 42
random.seed(seed)
np.random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)

torch.manual_seed(seed)            
torch.cuda.manual_seed(seed)       
torch.cuda.manual_seed_all(seed)

def set_random_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def collate_molgraphs(data):
    assert len(data[0]) in [5, 6], \
        'Expect the tuple to be of length 3 or 4, got {:d}'.format(len(data[0]))
    if len(data[0]) == 5:
        smiles, graphs,trans,finger, labels = map(list, zip(*data))
        masks = None
    else:
        smiles, graphs, trans,finger,labels, masks = map(list, zip(*data))

    bg = dgl.batch(graphs)
    bg.set_n_initializer(dgl.init.zero_initializer)
    bg.set_e_initializer(dgl.init.zero_initializer)
    labels = torch.stack(labels, dim=0)

    if masks is None:
        masks = torch.ones(labels.shape)
    else:
        masks = torch.stack(masks, dim=0)
    return smiles, bg,trans,finger, labels, masks

atom_featurizer = AttentiveFPAtomFeaturizer(atom_data_field='hv')
bond_featurizer = AttentiveFPBondFeaturizer(bond_data_field='he')
n_feats = atom_featurizer.feat_size('hv')
e_feats = bond_featurizer.feat_size('he')


def load_data(data,name,load,path_trans,path_finger):
 
    dataset = MoleculeCSVDataset(data,
                                 smiles_to_graph=smiles_to_bigraph,
                                 node_featurizer=atom_featurizer,
                                 edge_featurizer= bond_featurizer,
                                 smiles_column='Smiles',
                                 cache_file_path=str(name)+'_dataset.bin',
                                 task_names=['Tg/℃'],
                                 load=load,init_mask=True,n_jobs=1,
                                 path_trans=path_trans,
                                 path_finger = path_finger
                            )
    return dataset

path1 = pd.read_excel('data/dataset-Tg.xlsx')
#path1 = pd.read_excel('scaler_λcutoff_nm.xlsx')
path_trans = 'data/dataset-Tg.npy'
#path_finger = 'data/dataset-λcut-off.txt'
path_finger = 'datasetA-Tg_Smiles_X_QC_XUMAP_fing.xlsx'

datasets = load_data(path1,'Tg',True,path_trans=path_trans,path_finger=path_finger)

train_size = int(0.7 * len(datasets))
test_size = len(datasets) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(datasets, [train_size, test_size])

train_loader = DataLoader(datasets, batch_size=30,shuffle=True,
                          collate_fn=collate_molgraphs)
val_loader = DataLoader(test_dataset, batch_size=30,shuffle=True,
                          collate_fn=collate_molgraphs)



def run_a_train_epoch(n_epochs, epoch, model, data_loader, loss_criterion, optimizer):
    model.train()
    losses = []
    train_meter = Meter()
   
    n_max =480
    n_min =70
    n_mean =274.9997459
    for batch_id, batch_data in enumerate(data_loader):

        smiles, bg, trans,finger,labels, masks = batch_data
      
        bg=bg.to(device)
        trans = torch.tensor(np.array(trans)).to(device)
        finger = torch.tensor(np.array(finger)).to(device)
        labels = labels.to(device)
        labels = (labels-n_mean)/(n_max-n_min)
        masks = masks.to(device)
        n_feats = bg.ndata.pop('hv').to(device)
        e_feats = bg.edata.pop('he').to(device)
       
        prediction,embedding = model(bg, n_feats, e_feats,trans,finger)
       
        if epoch == n_epochs-1:
            print('Embedding data：',embedding.reshape(embedding.shape[0],4,-1)[0])
        numpy_data = embedding.reshape(embedding.shape[0],4,-1)[0].detach().cpu().numpy()

       
        np.savetxt('embedding_data.txt', numpy_data)
       
        loss = (loss_criterion(prediction, labels) * (masks != 0).float()).mean()
        optimizer.zero_grad()
      
        loss.backward()
        optimizer.step()
        train_meter.update(prediction, labels, masks)
        losses.append(loss.data.item())

    total_r2 = np.mean(train_meter.compute_metric('r2'))
    total_loss = np.mean(losses)
 
    if epoch % 10 == 0:
        print('epoch {:d}/{:d}, training_r2 {:.4f}, training_loss {:.4f}'.format(epoch + 1, n_epochs, total_r2,total_loss))


    return total_r2, total_loss

def run_an_eval_epoch(n_epochs, model, data_loader,loss_criterion, best_score):
    model.eval()
    val_losses=[]
    eval_meter = Meter()
    n_max = 480
    n_min = 70
    n_mean = 274.9997459
 
    with torch.no_grad():
        for batch_id, batch_data in enumerate(data_loader):

            smiles, bg, trans,finger,labels, masks = batch_data
            bg = bg.to(device)
            trans = torch.tensor(np.array(trans)).to(device)
            finger = torch.tensor(np.array(finger)).to(device)
            labels = labels.to(device)
            labels = (labels - n_mean) / (n_max - n_min)
            masks = masks.to(device)
            n_feats = bg.ndata.pop('hv').to(device)
            e_feats = bg.edata.pop('he').to(device)
            vali_prediction ,vali_embedding= model(bg, n_feats, e_feats,trans,finger)
            val_loss = (loss_criterion(vali_prediction, labels) * (masks != 0).float()).mean()
            val_loss=val_loss.detach().cpu().numpy()
            val_losses.append(val_loss)
            eval_meter.update(vali_prediction, labels, masks)
        total_score = np.mean(eval_meter.compute_metric('r2'))
        total_loss = np.mean(val_losses)
     
        if total_score > best_score:
            best_score = total_score
            torch.save(model.state_dict(), 'best_ffnn_model_Tg.pt')
            #print('Best data',best_score)
    return total_score, total_loss,best_score



model = Net().to(device)

model.load_state_dict(torch.load('best_ffnn_model_Tg.pt'))

loss_fn = nn.MSELoss(reduction='none')

optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay = 0.000001
                             )
# stopper = EarlyStopping(mode='higher', patience=20)

best_score = -np.inf
n_epochs = 501
for e in range(n_epochs):
    score = run_a_train_epoch(n_epochs, e, model, train_loader, loss_fn, optimizer)
    val_score0,val_score1,best_score = run_an_eval_epoch(n_epochs, model, val_loader,loss_fn,best_score)
    # early_stop = stopper.step(val_score[0], model)
    if e % 10 == 0:
        print('epoch {:d}/{:d}, validation {} {:.4f}, validation {} {:.4f}'.format(
        e + 1, n_epochs, 'r2', val_score0, 'loss', best_score,
        'r2'))



if __name__ == '__main__':
  

    torch.save(model.state_dict(), 'TM.pt')

