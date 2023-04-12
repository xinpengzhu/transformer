import torch
import torch.nn as nn
from layers.multi_head_attention import MultiHeadAttention
from layers.position_wise_feed_forward import PositionwiseFeedForward

class EncoderLayer(nn.Module):
    def __init__(self, d_model, d_ff, head, p_dropout=0.1):
        super().__init__()
        self.attn = MultiHeadAttention(d_model, head)
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(p=p_dropout)
        
        self.ffn = PositionwiseFeedForward(d_model, d_ff, p_dropout)
        self.layer_norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(p=p_dropout)
        
    # def forward(self, x, mask):
    #     norm1_x = self.layer_norm1(x)
    #     x = self.attn(norm1_x, norm1_x, norm1_x, mask)
    #     x = x + self.dropout1(x)
    #     norm2_x = self.layer_norm2(x)
    #     x = self.ffn(norm2_x)
    #     return x + self.dropout2(x)

    def forward(self, x, mask):
        x = self.layer_norm1(x + self.dropout1(self.attn(x, x, x, mask)))
        return self.layer_norm2(x + self.dropout2(self.ffn(x)))

class Encoder(nn.Module):
    def __init__(self, d_model, d_ff, head, N, p_dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([EncoderLayer(d_model, d_ff, head, p_dropout) for _ in range(N)])
    
    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return x