import re
import numpy as np
from tqdm import tqdm 

import sentencepiece as spm

import torch
import torch.nn as nn
import torch.optim as optim
from torchtext.legacy import data, datasets
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from sklearn.metrics import accuracy_score, precision_score, recall_score

torch.manual_seed(2018114018)
torch.backends.cudnn.deterministic = True

BATCH_SIZE = 64
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

TEXT = data.Field(tokenize = 'spacy', tokenizer_language = 'en_core_web_sm')
LABEL = data.LabelField(dtype = torch.float)

train_data, test_data = datasets.IMDB.splits(TEXT, LABEL)

test_text, test_labels = zip(*[[pair['text'],pair['label']] for pair in [vars(v) for v in test_data.examples]])
train_text, train_labels = zip(*[[pair['text'],pair['label']] for pair in [vars(v) for v in train_data.examples]])

def clean(s):
    s = s.strip()
    s = re.sub(r'<[^>]*>( )*[a-z]*( )*<[^>]*>',' ',s)
    s = re.sub(r'[ ]+',' ',re.sub(r'[^a-zA-Z ]',' ',s))
    return s.strip()

train_text_raw = [clean(' '.join(s)) for s in train_text]
test_text_raw = [clean(' '.join(s)) for s in test_text]

raw_text = train_text_raw + test_text_raw
with open('raw_text.txt','w') as f:
    f.writelines([l+'\n' for l in raw_text])

spm.SentencePieceTrainer.train('--input=raw_text.txt --model_prefix=imdb --vocab_size=5000')
sp = spm.SentencePieceProcessor()
sp.load('imdb.model')

MAX_SEQ_LEN = 64

def fix_len(s):
    if len(s) < MAX_SEQ_LEN:
        s = s*((MAX_SEQ_LEN//len(s))+1)
    return s[:MAX_SEQ_LEN]

train_text_cleaned = [fix_len(sp.encode_as_ids(s)) for s in tqdm(train_text_raw)]

test_text_cleaned = [fix_len(sp.encode_as_ids(s)) for s in tqdm(test_text_raw)]

train_labels_proced = [1.0 if v=='pos' else 0.0 for v in train_labels]
test_labels_proced = [1.0 if v=='pos' else 0.0 for v in test_labels]

class ImdbDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels

    def __len__(self):
        return len(self.texts)
        
    def __getitem__(self, idx):
        return np.array(self.texts[idx]), int(self.labels[idx])
    
train_iter = DataLoader(dataset=ImdbDataset(train_text_cleaned,train_labels_proced),batch_size=BATCH_SIZE,shuffle=True)
test_iter = DataLoader(dataset=ImdbDataset(test_text_cleaned,test_labels_proced),batch_size=BATCH_SIZE,shuffle=True)

class RNNClassifier(nn.Module):
    def __init__(self,vocab_size,input_dim=64,hidden_dim=128,n_classes=2,dropout=0.3,num_layers=1,bidirectional=False):
        super(RNNClassifier, self).__init__()
        self.embeddings = nn.Embedding(vocab_size,input_dim)
        self.rnn = nn.RNN(input_size=input_dim,
                          hidden_size=hidden_dim,
                          num_layers=num_layers,
                          bidirectional=bidirectional,
                          dropout=dropout,
                          batch_first=True)
        self.fc = nn.Linear(hidden_dim,n_classes)
        self.dropout = nn.Dropout(p=dropout)
        self.softmax = nn.Softmax(dim=-1)
        
    def forward(self,s):
        s = self.dropout(self.embeddings(s))
        out, h = self.rnn(s)
        x = self.fc(h)
        x = self.softmax(x)
        return x[0]

def train(train_iter,model,optimizer):
    
    loss_list = []
    pred_list = []
    label_list = []
    
    model.train()
    loss_fn = nn.CrossEntropyLoss().to(device)

    for text, label in train_iter:
        text = torch.LongTensor(text).to(device)
        label = torch.LongTensor(label).to(device)

        pred = model(text)
        
        loss = loss_fn(pred,label)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        loss_list.append(loss.item())
        
        pred_list.extend(pred.cpu().detach().numpy())
        label_list.extend(label.cpu().detach().numpy())
    
    return np.mean(loss_list), pred_list, label_list

def evaluate(test_iter):
    
    loss_list = []
    pred_list = []
    label_list = []
    
    model.eval()
    loss_fn = nn.CrossEntropyLoss().to(device)

    with torch.no_grad():
        for text, label in test_iter:
            text = torch.LongTensor(text).to(device)
            label = torch.LongTensor(label).to(device)

            pred = model(text)

            loss = loss_fn(pred,label)
            loss_list.append(loss.item())

            pred_list.extend(pred.cpu().detach().numpy())
            label_list.extend(label.cpu().detach().numpy())

    return np.mean(loss_list), pred_list, label_list
        

model = RNNClassifier(vocab_size=5000,
                     input_dim=256,
                     hidden_dim=256,
                     num_layers=2).to(device)
optimizer = optim.Adam(model.parameters(),lr=5e-4,weight_decay=5e-4)

epochs = 100
patience = 10

for e in range(epochs):
    train_loss, train_pred_list, train_label_list = train(train_iter,model,optimizer)
    eval_loss, eval_pred_list, eval_label_list = evaluate(test_iter) 

    train_pred_list = np.argmax(train_pred_list,axis=1); train_acc = accuracy_score(train_label_list,train_pred_list)
    eval_pred_list = np.argmax(eval_pred_list,axis=1); 
    test_acc = accuracy_score(eval_label_list,eval_pred_list)
    test_pre = precision_score(eval_label_list,eval_pred_list)
    test_rec = recall_score(eval_label_list,eval_pred_list)
    print(f">> epoch {e:3.0f} | train loss: {train_loss:6.4f} | test loss: {eval_loss:6.4f} | train acc: {train_acc:6.4f} | test acc: {test_acc:6.4f} | test pre: {test_pre:6.4f} | test rec: {test_rec:6.4f}")