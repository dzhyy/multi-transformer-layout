import torch
from torch import nn
from torch.nn import Parameter
import torch.nn.functional as F
import copy

# Code adapted from the fairseq repo.

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

class MultiheadAttention(nn.Module):
    """Multi-headed attention.
    See "Attention Is All You Need" for more details.
    """

    def __init__(self, embed_dim, num_heads, attn_dropout=0.,
                 bias=True):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.attn_dropout = attn_dropout
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"
        self.scaling = self.head_dim ** -0.5

        self.linear = _get_clones(nn.Linear(embed_dim, embed_dim), 4)


    def forward(self, query, key, value, attn_mask=None):
        # [len,3,30]
        """Input shape: Time x Batch x Channel
        Self-attention can be implemented by passing in the same arguments for
        query, key and value. Timesteps can be masked by supplying a T x T mask in the
        `attn_mask` argument. Padding elements can be excluded from
        the key by passing a binary ByteTensor (`key_padding_mask`) with shape:
        batch x src_len, where padding elements are indicated by 1s.
        """

        tgt_len, bsz, embed_dim = query.size()
        src_len = key.size(0)
        assert embed_dim == self.embed_dim
        assert key.size() == value.size()

        q, k, v = [l(x) for l, x in zip(self.linear, (query, key, value))] # [len,3,30] -> [len,3,30]

        q = q.transpose(0, 1).reshape(bsz, tgt_len, self.num_heads, self.head_dim).transpose(1, 2)  # [bn,len,30]->[bn, nheads, len, hdim]
        k = k.transpose(0, 1).reshape(bsz, -1, self.num_heads, self.head_dim).transpose(1, 2)       # [bn,len,30]->[bn, nheads, len, hdim]
        v = v.transpose(0, 1).reshape(bsz, -1, self.num_heads, self.head_dim).transpose(1 ,2)       # [bn,len,30]->[bn, nheads, len, hdim]

        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scaling # [bn, nheads, len, len]
        assert list(attn_weights.size()) == [bsz , self.num_heads, tgt_len, src_len]

        if attn_mask is not None:
            try:
                attn_weights.masked_fill(attn_mask.unsqueeze(1)==False, float('-inf')) # [bn, nheads, len, len] mask.unsqueeze(1): [bn,1,len,len] or [bn,1,1,len] value = 0 or -inf
            except:
                print(attn_weights.shape)
                print(attn_mask.unsqueeze(0).shape)
                assert False
                
        attn_weights = F.softmax(attn_weights.float(), dim=-1).type_as(attn_weights)
        attn_weights = F.dropout(attn_weights, p=self.attn_dropout, training=self.training)

        attn = torch.matmul(attn_weights, v)
        assert list(attn.size()) == [bsz, self.num_heads, tgt_len, self.head_dim]

        attn = attn.transpose(1,2).reshape(bsz, tgt_len, embed_dim).transpose(0,1) # attn.transpose(1,2):''[bn, len, nheads, d_k]'' final: [len, bn, d_model]
        
        attn = self.linear[-1](attn)

        # average attention weights over heads
        attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
        attn_weights = attn_weights.sum(dim=1) / self.num_heads
        return attn, attn_weights # [25,3,30]

    def in_proj_qkv(self, query):
        return self._in_proj(query).chunk(3, dim=-1)

    def in_proj_kv(self, key):
        return self._in_proj(key, start=self.embed_dim).chunk(2, dim=-1)

    def in_proj_q(self, query, **kwargs):
        return self._in_proj(query, end=self.embed_dim, **kwargs)

    def in_proj_k(self, key):
        return self._in_proj(key, start=self.embed_dim, end=2 * self.embed_dim)

    def in_proj_v(self, value):
        return self._in_proj(value, start=2 * self.embed_dim)

    def _in_proj(self, input, start=0, end=None, **kwargs):
        weight = kwargs.get('weight', self.in_proj_weight)
        bias = kwargs.get('bias', self.in_proj_bias)
        weight = weight[start:end, :]
        if bias is not None:
            bias = bias[start:end]
        return F.linear(input, weight, bias)
