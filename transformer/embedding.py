import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_len):
        super().__init__()
        pe = torch.zeros(max_seq_len, d_model)
        pos = torch.arange(max_seq_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000) / d_model))
        pe[:, ::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe)
    
    def forward(self, x):
        seq_len = x.size(1)
        return self.pe[:seq_len]

class TransformerEmbedding(nn.Module):
    def __init__(self, vocab_size, d_model, max_seq_len, p_dropout):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, d_model)
        self.pos = PositionalEncoding(d_model, max_seq_len)
        self.drop = nn.Dropout(p=p_dropout)
        

    def forward(self, x):
        emb = self.emb(x)
        pos = self.pos(x)
        return self.drop(emb + pos)