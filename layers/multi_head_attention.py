import torch.nn as nn
import torch
import math
import torch.nn.functional as F

def scaledotproductattention(q, k, v, mask=None):
    d_k = k.size(-1)
    score = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(d_k)
    if mask is not None:
        score = score.masked_fill(mask == 0, -1e9)
    attn = F.softmax(score, dim=-1)
    return torch.matmul(attn, v), attn

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, head) -> None:
        super().__init__()
        assert d_model % head == 0 
        self.head = head
        self.d_k = d_model // head
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.fc = nn.Linear(d_model, d_model)
    
    def forward(self, q, k, v, mask):
        batch_size = q.size(0)
        
        q = self.w_q(q).view(batch_size, -1, self.head, self.d_k).transpose(1,2)

        k = self.w_k(k).view(batch_size, -1, self.head, self.d_k).transpose(1,2)

        v = self.w_v(v).view(batch_size, -1, self.head, self.d_k).transpose(1,2)

        out, attn = scaledotproductattention(q, k, v, mask)

        out = out.transpose(1, 2).contiguous().view(batch_size, -1, self.head * self.d_k)

        return self.fc(out)