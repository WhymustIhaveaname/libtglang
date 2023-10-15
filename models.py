#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

import time
import traceback
import os
import math
import torch
import torch.nn as nn

LOGLEVEL = {0:"DEBUG",1:"INFO",2:"WARN",3:"ERR",4:"FATAL"}
LOGFILE  = "net.log"

def log(*msg,l=1,end="\n",logfile=LOGFILE):
    msg=", ".join(map(str,msg))
    st=traceback.extract_stack()[-2]
    lstr=LOGLEVEL[l]
    now_str="%s %03d"%(time.strftime("%y/%m/%d %H:%M:%S",time.localtime()),math.modf(time.time())[0]*1000)
    perfix="%s [%s,%s:%03d]"%(now_str,lstr,st.name,st.lineno)
    if l<3:
        tempstr="%s %s%s"%(perfix,str(msg),end)
    else:
        tempstr="%s %s:\n%s%s"%(perfix,str(msg),traceback.format_exc(limit=5),end)
    print(tempstr,end="")
    if l>=1:
        with open(logfile,"a") as f:
            f.write(tempstr)

class TransformerModel(nn.Module):
    def __init__(self,lenvocab,numlang):
        super().__init__()
        self.model_type = 'Transformer'
        self.ntoken_src = lenvocab
        self.ntoken_tgt = numlang
        self.d_model = 128

        self.src_encoder = nn.Embedding(self.ntoken_src, self.d_model)
        self.pos_encoder = PositionalEncoding(self.d_model)
        # batch_first (bool) – If True, then the input and output tensors are provided as (batch, seq, feature)
        self.transformer = nn.TransformerEncoderLayer(d_model=self.d_model, nhead=4, dim_feedforward=512, batch_first=True)
        self.generator   = nn.Linear(self.d_model, self.ntoken_tgt)

        log("there are %d vocabs"%(self.ntoken_src))

    def forward(self,src):
        bsf = self.src_encoder(src) * math.sqrt(self.d_model)
        bsf = self.pos_encoder(bsf)
        out = self.transformer(bsf)
        out[src==0] = float('nan')
        out = out.nanmean(dim=-2)
        return self.generator(out)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 4096):
        super().__init__()

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return x