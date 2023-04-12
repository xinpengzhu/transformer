import torch
import torch.nn as nn
from layers.multi_head_attention import MultiHeadAttention
from layers.position_wise_feed_forward import PositionwiseFeedForward

class DecoderLayer(nn.Module):
    def __init__(self, d_model, d_ff, head, p_dropout=0.1):
        super().__init__()
        self.attn = MultiHeadAttention(d_model, head)
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(p=p_dropout)

        self.enc_dec_attn = MultiHeadAttention(d_model, head)
        self.layer_norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(p=p_dropout)

        self.ffn = PositionwiseFeedForward(d_model, d_ff)
        self.layer_norm3 = nn.LayerNorm(d_model)
        self.dropout3 = nn.Dropout(p=p_dropout)
    
    # def forward(self, x, memery, src_mask, tgt_mask):
    #     norm1_x = self.layer_norm1(x)
    #     x = x + self.dropout1(self.attn(norm1_x, norm1_x, norm1_x, tgt_mask))
    #     norm2_x = self.layer_norm2(x)
    #     x = x + self.dropout2(self.enc_dec_attn(norm2_x, memery, memery, src_mask))
    #     norm3_x = self.layer_norm3(x)
    #     return x + self.dropout3(self.ffn(norm3_x))

    def forward(self, x, memery, src_mask, tgt_mask):
        x = self.layer_norm1(x + self.dropout1(self.attn(x, x, x, tgt_mask)))
        x = self.layer_norm2(x + self.dropout2(self.enc_dec_attn(x, memery, memery, src_mask)))
        return self.layer_norm3(x + self.dropout3(self.ffn(x)))

class Decoder(nn.Module):
    def __init__(self, trg_vocab_size, d_model, d_ff, head, N, p_dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([DecoderLayer(d_model, d_ff, head, p_dropout) for _ in range(N)])
        self.fc = nn.Linear(d_model, trg_vocab_size)

    def forward(self, x, memery, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, memery, src_mask, tgt_mask)
        out = self.fc(x)
        return out
