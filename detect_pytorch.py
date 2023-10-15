#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

import os
import re
import math
import ctypes
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import TensorDataset,DataLoader
from torchtext.datasets import WikiText2
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

from models import *

torch.manual_seed(2023)

padlen = 2048
languages = {'human':0,'bash':83,'clojure':16,'cpp':21,'csharp':23,'css':24,
             'fsharp':36,'go':39,'java':48,'js':49,'php':68,
             'powershell':70,'python':73,'ruby':78,'scala':81,'sql':86,
             'xml':98}
languages2 = {j:i for i,(j,k) in enumerate(languages.items())}
languages3 = {languages2[i]:languages[i] for i in languages}

log(languages2,languages3)

def load_data(dir='./data/train'):
    l = []
    for lang in os.listdir(dir):
        langidx = languages2[lang]
        langdir = os.path.join(dir,lang)
        for i,file in enumerate(os.listdir(langdir)):
            if i>50:
                break
            file = os.path.join(langdir,file)
            with open(file,'r') as f:
                code = f.read().strip()
            code = re.sub(r'[^\x00-\x7F]+','<unk>',code)
            if len(code)<padlen:
                l.append((langidx,code))
            else:
                l += [(langidx,code[i:i+padlen]) for i in range(0,len(code),padlen)]
    return l

trainiter = load_data()
tokenizer = get_tokenizer("revtok")
#tokenizer = lambda x: x.split()
vocab = build_vocab_from_iterator(map(tokenizer,(j for i,j in trainiter)), specials=['<pad>','<unk>'], max_tokens=1000)
vocab.set_default_index(vocab['<unk>'])

log(vocab.get_itos())

def data_process(textiter):
    data = [torch.tensor(vocab(tokenizer(j)), dtype=torch.long) for i,j in textiter]
    data = pad_sequence(data,batch_first=True,padding_value=vocab['<pad>'])
    log('data shape: %s'%(data.shape,))
    data = TensorDataset(data, torch.tensor([i for i,j in textiter],dtype=torch.long))
    data = DataLoader(dataset=data,batch_size=16,shuffle=True,drop_last=True)
    return data

traindata = data_process(trainiter)

assert vocab['<pad>']==0, 'because this 0 is hardcoded in models.py'
model = TransformerModel(len(vocab),len(languages))

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

def train(epoch=-1):
    model.train()
    accloss = []
    for code,target in tqdm(traindata):
        output  = model(code)
        loss    = criterion(output, target)
        accloss.append(loss.item())

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
    optimizer.step()

    accloss = torch.tensor(accloss)
    log("epoch %3d, loss %.4f"%(epoch,accloss.mean().item()))

def train_epochs():
    for epoch in range(5):
        train(epoch)

    export_model()

def export_model(save_name='detect_model.pt'):
    # torch.save(tokenizer, 'tokenizer.pt')
    # torch.save(vocab, 'vocab.pt')
    model_scripted = torch.jit.script(model) # Export to TorchScript
    model_scripted.save(save_name) # Save
    log("saved to %s"%(save_name))

def detect(text: str) -> int:
    print(text)
    text = torch.tensor(vocab(tokenizer(text))).unsqueeze(0)
    print(text)
    lang = model(text).squeeze()
    print(lang)
    _,lang = torch.max(lang,dim=-1)
    return languages3[lang.item()]

def pre_process(text: str) -> int:
    return torch.tensor(vocab(tokenizer(text))).unsqueeze(0)

if __name__=="__main__":
    torch.save(pre_process,'pre_process.pt')
    #train_epochs()
    #print(detect("print(torch.max(lang,dim=-1))"))
    # detect_func = torch.jit.script(tokenizer)
    # detect_func.save("tokenizer.pt")
