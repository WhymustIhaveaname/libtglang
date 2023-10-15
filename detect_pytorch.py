#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

import math
import ctypes
import torch
import torch.nn as nn
from torchtext.datasets import WikiText2
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torch.nn.utils.rnn import pad_sequence

torch.manual_seed(2023)

train_iter = WikiText2(split='train')
for i in train_iter:
    print(i)
tokenizer = get_tokenizer('basic_english')
vocab = build_vocab_from_iterator(map(tokenizer, train_iter), specials=['<unk>'])
vocab.set_default_index(vocab['<unk>'])


class TransformerModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model_type = 'Transformer'
        self.ntoken_src = 128
        self.ntoken_tgt = 100 # number of different programming languages
        self.d_model = 512

        self.src_encoder = nn.Embedding(self.ntoken_src, self.d_model)
        self.pos_encoder = PositionalEncoding(self.d_model)
        # batch_first (bool) – If True, then the input and output tensors are provided as (batch, seq, feature)
        self.transformer = nn.TransformerEncoderLayer(d_model=self.d_model, nhead=8, batch_first=True)
        self.generator   = nn.Linear(self.d_model, self.ntoken_tgt)

    def forward(self,src):
        src = torch.rand(32, 10, 512)
        # src = self.src_encoder(src) * math.sqrt(self.d_model)
        # src = self.pos_encoder(src)
        out = self.transformer(src)
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

def export_model(save_name='detect_model.pt'):
    model = TransformerModel()
    print(model.forward(None).shape)
    model_scripted = torch.jit.script(model) # Export to TorchScript
    model_scripted.save(save_name) # Save
    print("saved to %s"%(save_name))

if __name__=="__main__":
    #export_model()
    # TglangLanguage = ctypes.CDLL('./tglang.h')
    # print(TglangLanguage.TGLANG_LANGUAGE_PYTHON)
    pass