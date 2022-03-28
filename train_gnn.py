import sys
import numpy as np
import random
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.linalg import matrix_power as mat_pow

from gnn import GCN, GSage

import warnings
warnings.filterwarnings("ignore")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

torch.manual_seed(2018114018)
torch.backends.cudnn.deterministic = True


def dataloader(cites_path,content_path):

    feats, labels = [], []
    id2node, node2id = {}, {}
    id2label, label2id = {}, {}
    with open(content_path,'r') as f:
        for i, l in enumerate(f):
            line = l.strip().split()
            feats.append([float(v) for v in line[1:-1]])
            line[0] = str(line[0]).strip()
            id2node[i] = line[0]
            node2id[line[0]] = i
            label = line[-1]
            if label not in label2id:
                id2label[len(id2label)] = label
                label2id[label] = len(label2id)
            labels.append(label2id[label])

    feats = np.array(feats)

    N, Fc = feats.shape
    L = len(label2id)

    for i, l in enumerate(labels):
        class_set = [0.0]*L
        class_set[l-1] = 1.0
        labels[i] = class_set
    labels = np.array(labels,dtype=np.int64)



    adj = np.zeros((N,N))
    with open(cites_path,'r') as f:
        for l in f:
            u, v = l.strip().split()
            if u in node2id and v in node2id:
                adj[node2id[u]][node2id[v]] = 1.0

    adj = adj + np.eye(N)    # A-cap = A + I
    
    deg = np.zeros_like(adj)
    adj_deg = np.sum(adj,axis=1)
    for i in range(deg.shape[0]):
        deg[i][i] = adj_deg[i]

    print(f"total nodes (N): {N}")
    print(f"total feats (Fc): {Fc}")
    print(f"total class labels (L): {L}")
    print()
    print(f"adj built (NxN): {adj.shape}")
    print(f"feats built (NxFc): {feats.shape}")
    print(f"labels built (NxL): {labels.shape}")
    print()

    test_size = 0.3
    node_list = random.sample(id2node.keys(),k=len(id2node.keys()))
    split_id = int(len(node_list)*test_size)
    test_ids, train_ids = node_list[:split_id], node_list[split_id:]

    print(f"len(node_list) (N) {len(node_list)}")
    print(f"len(train_ids): {len(train_ids)}")
    print(f"len(test_ids):  {len(test_ids)}")

    feats = torch.Tensor(feats).to(device)
    adj = torch.Tensor(adj).to(device)
    deg = torch.Tensor(deg).to(device)
    labels = torch.Tensor(labels).long().to(device)
    
    return (feats, adj, deg, labels), (train_ids, test_ids), (id2label, label2id, id2node, node2id), (N, Fc, L)



def train(cat,feats,adj,deg,labels,split_ids,model,optimizer):
    
    train_ids, test_ids = split_ids
    
    model.train()
    preds = model(feats,adj)
    preds = F.log_softmax(preds,dim=1)
    loss = F.cross_entropy(preds[train_ids],labels[:,cat][train_ids])
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    _labels = labels.cpu().detach().numpy()
    _preds = preds.cpu().detach().numpy()

    train_logs = {
        'train_loss':loss.item(),
        'train_acc':accuracy_score(_labels[:,cat][train_ids],np.argmax(_preds,axis=1)[train_ids]==cat),
        'train_pre':precision_score(_labels[:,cat][train_ids],np.argmax(_preds,axis=1)[train_ids]==cat),
        'train_rec':recall_score(_labels[:,cat][train_ids],np.argmax(_preds,axis=1)[train_ids]==cat),
    }
    return train_logs
    
def test(cat,feats,adj,deg,labels,split_ids,model):
    
    train_ids, test_ids = split_ids
    
    model.eval()
    with torch.no_grad():
        preds = model(feats,adj)
        preds = F.log_softmax(preds,dim=1)
        loss = F.cross_entropy(preds[test_ids],labels[:,cat][test_ids])
        _labels = labels.cpu().detach().numpy()
        _preds = preds.cpu().detach().numpy()
        
        test_logs = {
            'test_loss':loss.item(),
            'test_acc':accuracy_score(_labels[:,cat][test_ids],np.argmax(_preds,axis=1)[test_ids]==cat),
            'test_pre':precision_score(_labels[:,cat][test_ids],np.argmax(_preds,axis=1)[test_ids]==cat),
            'test_rec':recall_score(_labels[:,cat][test_ids],np.argmax(_preds,axis=1)[test_ids]==cat),
        }
        return test_logs


def fit_and_evaluate(
    adj,feats,deg,labels,
    split_ids,
    node_label_mappings,
    in_dim,out_dim,
    epochs = 50,
    lr = 5e-3,
    dropout = 0.2,
    weight_decay = 5e-4,
    activation = 'relu',
    gnn = 'gcn',
    mat_norm = 'row',
    n_layers = 1,
    hidden_dim = None,
    adj_norm = None,
    device = 'cuda'
    ):

    train_ids, test_ids = split_ids
    id2label, label2id, id2node, node2id = node_label_mappings

    #     https://math.stackexchange.com/questions/3035968/interpretation-of-symmetric-normalised-graph-adjacency-matrix
    if adj_norm == 'row':
        adj = mat_pow(deg,n=-1) @ adj
    if adj_norm == 'col':
        adj = adj @ mat_pow(deg,n=-1)
    if adj_norm == 'sym':
        deg_h = mat_pow(deg,n=-1)
        adj = (deg_h @ adj) @ deg_h

    hidden_dim = in_dim if not hidden_dim else hidden_dim
    feat_dim = feats.shape[1]

    if gnn=='gcn':
        model = GCN(in_dim=in_dim,feat_dim=feat_dim,out_dim=out_dim,hidden_dim=hidden_dim,n_layers=n_layers,activation=F.relu,dropout=dropout)
        
    elif gnn=='gsage':
        model = GSage(in_dim=in_dim,feat_dim=feat_dim,out_dim=out_dim,hidden_dim=hidden_dim,n_layers=n_layers,activation=F.relu,dropout=dropout)

    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=lr,
                                 weight_decay=weight_decay)

    logs = []
    agg_logs = {}

    for cat,label in list(id2label.items()):

        cat_logs = {k:[] for k in [f"{splt}_{metric}" for splt in ['train','test'] for metric in ['loss','acc','pre','rec']]}

        for e in range(epochs):
            train_logs = train(cat,feats,adj,deg,labels,split_ids,model,optimizer)
            test_logs = test(cat,feats,adj,deg,labels,split_ids,model) 

            for k,v in train_logs.items():
                cat_logs[k].append(v)

            for k,v in test_logs.items():
                cat_logs[k].append(v)

        best_epoch = np.argmax(cat_logs['test_acc'])
        best_metrics = {k:cat_logs[k][best_epoch] for k in cat_logs.keys()}

        cat_contrib = float(sum(labels[:,cat]==1)/labels.shape[0])
        for k in cat_logs.keys():
            if k not in agg_logs:
                agg_logs[k] = 0
            agg_logs[k] += cat_contrib * float(best_metrics[k])

        logs.append(cat_logs)

    return agg_logs, logs


if __name__ == "__main__":

    gnn = sys.argv[1] if len(sys.argv)==2 else 'gcn'
    if gnn not in ['gnn','gsage']:
        raise Exception("Invalid GNN type specified. Choose from ['gnn','gsage']")

    print(f"\nLoading data\n")
    (feats, adj, deg, labels), (train_ids, test_ids), (id2label, label2id, id2node, node2id), (N, Fc, L) = dataloader(cites_path='data/citeseer/citeseer.cites',content_path='data/citeseer/citeseer.content')

    n_layers_list = [1,2,3]
    adj_norm_list = ['row','col','sym']
    hidden_dim_list = [128,256,512]

    print(f"\nTraining [ {gnn} ]\n")

    for n_layers in n_layers_list:
        for adj_norm in adj_norm_list:
            for hidden_dim in hidden_dim_list:
                print(f"[ n_layers: {n_layers:2.0f} | adj_norm: {'non' if not adj_norm else adj_norm} | hidden_dim: {hidden_dim:5.0f} ]")
                agg_logs, logs = fit_and_evaluate(adj,feats,deg, labels,
                                                (train_ids, test_ids),
                                                (id2label, label2id, id2node, node2id),
                                                in_dim=Fc,out_dim=L,
                                                hidden_dim=hidden_dim,
                                                n_layers=n_layers,
                                                adj_norm=adj_norm,
                                                gnn=gnn,
                                                device=device)
                print(f">> testacc: {agg_logs['test_acc']:5.4f}")
                print()
                # if n_layers == 1:
                #     break

    print(f"Training and evaluation complete")