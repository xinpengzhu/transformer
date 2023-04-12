import torch
import torch.nn as nn
from encoder import Encoder
from decoder import Decoder
from embedding import TransformerEmbedding

def make_pad_mask(q, k, q_pad_idx, k_pad_idx):
    len_q, len_k = q.size(1), k.size(1)

    # batch_size x 1 x 1 x len_k
    k = k.ne(k_pad_idx).unsqueeze(1).unsqueeze(2)
    # batch_size x 1 x len_q x len_k
    k = k.repeat(1, 1, len_q, 1)

    # batch_size x 1 x len_q x 1
    q = q.ne(q_pad_idx).unsqueeze(1).unsqueeze(3)
    # batch_size x 1 x len_q x len_k
    q = q.repeat(1, 1, 1, len_k)

    mask = k & q
    return mask

def attn_dec_mask(shape):
    mask = torch.tril(torch.ones(shape)).to(torch.bool).cuda()
    return mask

class Transformer(nn.Module):
    def __init__(self, src_pad_idx, trg_pad_idx, d_model, src_vocab_size, trg_vocab_size, max_seq_len, d_ff, head, N, p_dropout=0.1):
        super().__init__()
        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        self.src_emb = TransformerEmbedding(src_vocab_size, d_model, max_seq_len, p_dropout)
        self.encoder = Encoder(d_model, d_ff, head, N, p_dropout)
        self.trg_emb = TransformerEmbedding(trg_vocab_size, d_model, max_seq_len, p_dropout)
        self.decoder = Decoder(trg_vocab_size, d_model, d_ff, head, N, p_dropout)

    def forward(self, src_seq, trg_seq):
        src_mask = make_pad_mask(src_seq, src_seq, self.src_pad_idx, self.src_pad_idx)
        trg_mask = make_pad_mask(trg_seq, trg_seq, self.trg_pad_idx, self.trg_pad_idx)
        trg_mask = trg_mask & attn_dec_mask(trg_mask.shape)

        trg_src_mask = make_pad_mask(trg_seq, src_seq, self.trg_pad_idx, self.src_pad_idx)
        
        src_emb = self.src_emb(src_seq)
        src_enc = self.encoder(src_emb, src_mask)
        trg_emb = self.trg_emb(trg_seq)
        trg_dec = self.decoder(trg_emb, src_enc, trg_src_mask, trg_mask)
        return trg_dec