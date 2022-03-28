import torch
import torch.nn as nn
import torch.nn.functional as F

import warnings
warnings.filterwarnings("ignore")

torch.manual_seed(2018114018)
torch.backends.cudnn.deterministic = True

class GCNLayer(nn.Module):
    
    def __init__(self,in_dim,feat_dim,out_dim,
                 activation=F.relu,
                 dropout=0.2,
                 bias=True):
    
        super(GCNLayer, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.activation = activation if activation else None
        self.dropout = nn.Dropout(dropout) if dropout else None
        
        self.W = nn.Parameter(torch.FloatTensor(in_dim,out_dim))
        self.b = nn.Parameter(torch.FloatTensor(out_dim)) if bias else torch.zeros((out_dim))
        
        self.initialize_weights()
        
    def initialize_weights(self):
        for p in self.parameters():
            if len(p.size())==2:
                nn.init.xavier_uniform_(p)
            else:
                nn.init.constant_(p,float(0))
                
    def forward(self,feats,adj):
        feats = self.dropout(feats) if self.dropout else feats
        x = adj @ feats            # messaging
        x = x @ self.W + self.b    # aggregation
        x = self.activation(x) if self.activation else x
        return x


class GCN(nn.Module):
    
    def __init__(self,in_dim,feat_dim,out_dim,
                 hidden_dim=None,
                 n_layers=1,
                 activation=F.relu,
                 dropout=0.2,
                 bias=True):
    
        super(GCN, self).__init__()
        
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.feat_dim = feat_dim
        self.hidden_dim = hidden_dim if hidden_dim else in_dim
        self.n_layers = n_layers
        self.activation = activation
        self.dropout = dropout 
        self.bias = bias
        
        self.layers = nn.ModuleList()
        
        if self.n_layers == 1:
            self.layers.append(GCNLayer(in_dim=self.in_dim,feat_dim=self.feat_dim,out_dim=self.out_dim,activation=self.activation,dropout=self.dropout))
        else:
            self.layers.append(GCNLayer(in_dim=self.in_dim,feat_dim=self.feat_dim,out_dim=self.hidden_dim,activation=self.activation,dropout=self.dropout))
            self.layers.extend([GCNLayer(in_dim=self.hidden_dim,feat_dim=self.feat_dim,out_dim=self.hidden_dim,activation=self.activation,dropout=self.dropout) for _ in range(self.n_layers-2)])
            self.layers.append(GCNLayer(in_dim=self.hidden_dim,feat_dim=self.feat_dim,out_dim=self.out_dim,activation=self.activation,dropout=self.dropout)) 
                
    def forward(self,feats,adj):
        for i in range(self.n_layers):
            feats = self.layers[i](feats,adj)
        return feats


class GSageLayer(nn.Module):
    
    def __init__(self,in_dim,feat_dim,out_dim,
                 activation=F.relu,
                 dropout=0.2,
                 bias=True):
    
        super(GSageLayer, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.feat_dim = feat_dim
        self.activation = activation if activation else None
        self.dropout = nn.Dropout(dropout) if dropout else None
        self.bias = bias
        
        self.fc_neigh = nn.Linear(self.in_dim,self.out_dim,bias=self.bias)
        self.fc_self = nn.Linear(self.in_dim,self.out_dim,bias=self.bias)
        
        self.initialize_weights()
        
    def initialize_weights(self):
        for p in self.parameters():
            if len(p.size())==2:
                nn.init.xavier_uniform_(p)
            else:
                nn.init.constant_(p,float(0))
                
    def forward(self,feats,adj):
        feats = self.dropout(feats) if self.dropout else feats
        x = adj @ feats
        x_neigh = self.fc_neigh(x)
        x_self = self.fc_self(feats)
        x = x_neigh + x_self
        x = self.activation(x) if self.activation else x
        x = F.normalize(x,dim=1)
        return x

    
class GSage(nn.Module):
    
    def __init__(self,in_dim,feat_dim,out_dim,
                 hidden_dim=None,
                 n_layers=1,
                 activation=F.relu,
                 dropout=0.2,
                 bias=True):
    
        super(GSage, self).__init__()
        
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.feat_dim = feat_dim
        self.hidden_dim = hidden_dim if hidden_dim else in_dim
        self.n_layers = n_layers
        self.activation = activation
        self.dropout = dropout 
        self.bias = bias
        
        self.layers = nn.ModuleList()
        
        if self.n_layers == 1:
            self.layers.append(GSageLayer(in_dim=self.in_dim,feat_dim=self.feat_dim,out_dim=self.out_dim,activation=self.activation,dropout=self.dropout))
        else:
            self.layers.append(GSageLayer(in_dim=self.in_dim,feat_dim=self.feat_dim,out_dim=self.hidden_dim,activation=self.activation,dropout=self.dropout))
            self.layers.extend([GSageLayer(in_dim=self.hidden_dim,feat_dim=self.feat_dim,out_dim=self.hidden_dim,activation=self.activation,dropout=self.dropout) for _ in range(self.n_layers-2)])
            self.layers.append(GSageLayer(in_dim=self.hidden_dim,feat_dim=self.feat_dim,out_dim=self.out_dim,activation=self.activation,dropout=self.dropout)) 
                
    def forward(self,feats,adj):
        for i in range(self.n_layers):
            feats = self.layers[i](feats,adj)
        return feats