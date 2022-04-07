# encoding=utf-8
import math
import copy
from typing import Optional
import torch
from torch import Tensor
import torch.nn as nn
from torch.nn import (TransformerEncoder, TransformerDecoder, TransformerEncoderLayer, TransformerDecoderLayer)
import torch.nn.functional as F


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


##①##
# (n_batch,seq_len)-->(n_batch,seq_len,d_model)
class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, d_model: int):  # (32,56)-->(32,56,512),embedding_size = d_model != vocab_size
        super(TokenEmbedding, self).__init__()
        self.embedding = nn.Embedding(
            vocab_size,
            d_model,
        )
        self.d_model = d_model

    def forward(self, tokens: Tensor):
        return self.embedding(tokens.long()) * math.sqrt(self.d_model)


# 一开始值大(embed)，后面值小(attention)

##②##
class PositionalEncoding(nn.Module):
    def __init__(self, emb_size: int, dropout, maxlen: int = 5000):
        super(PositionalEncoding, self).__init__()
        den = torch.exp(- torch.arange(0, emb_size, 2) * math.log(10000) / emb_size)
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        pos_embedding = torch.zeros((maxlen, emb_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        pos_embedding = pos_embedding.unsqueeze(-2)

        self.dropout = nn.Dropout(dropout)
        self.register_buffer('pos_embedding', pos_embedding)

    def forward(self, token_embedding: Tensor):
        return self.dropout(token_embedding + self.pos_embedding[:token_embedding.size(0), :])


##③##
class EncoderDecoder(nn.Module):
    def __init__(self, encoder, decoder, src_embed, tgt_embed,
                 generator):  # src_embed=TokenEmbedding+PositionalEncoding
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed  # embedding model
        self.tgt_embed = tgt_embed
        self.generator = generator

    def forward(self, src, tgt, src_mask, tgt_mask):
        return self.decode(tgt, self.encode(src, src_mask), tgt_mask)
        # output shape (n_batch,seq_len,d_model)；不是(n_batch,seq_len,vocab)，generator保存在模型中供调用而非参与计算。如果要生成或计算损失，还要调用计算一下。
        # 维护decoder和encoder的输出形状统一??
        # 对于decoder，src_mask还有用吗?---有用，在尝试注意memory时用到

    def encode(self, src, src_mask):  # 为了embed和position info add
        return self.encoder(self.src_embed(src), src_mask)

    def decode(self, tgt, memory, tgt_mask):
        return self.decoder(self.tgt_embed(tgt), memory, tgt_mask)


class Encoder(nn.Module):
    def __init__(self, encoder_layer, N):  # N*encoder_layer
        super(Encoder, self).__init__()
        self.layers = _get_clones(encoder_layer, N)
        self.norm = LayerNorm(encoder_layer.size)

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)
        # 1*Norm1    encode final


class Decoder(nn.Module):
    def __init__(self, decoder_layer, N):  # N*decoder_layer
        super(Decoder, self).__init__()
        self.layers = _get_clones(decoder_layer, N)
        self.norm = LayerNorm(decoder_layer.size)

    def forward(self, x, memory, tgt_mask):
        for layer in self.layers:
            x = layer(x, memory, tgt_mask)
        return self.norm(x)
    # 1*norm    decoder final


class SublayerConnection(nn.Module):
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))
        # n*Norm
        # n*dropout      between layers


class EncoderLayer(nn.Module):
    def __init__(self, size, self_attn, fnn, dropout):  # encoder_layer = self_attention + add&norm + fnn + add&norm
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.fnn = fnn
        self.sublayer = _get_clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        # sublayer(x,f)要求x-输入->f。对于后者无法输入而已经得到值的情况下，使用lambda接受输入，返回这个值
        return self.sublayer[1](x, self.fnn)


class DecoderLayer(nn.Module):
    def __init__(self, size, self_atten, src_atten, fnn,
                 dropout):  # = self_attention + add&norm + memory_attention +add&norm + fnn + add&norm
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_atten
        self.src_attn = src_atten
        self.fnn = fnn
        self.sublayer = _get_clones(SublayerConnection(size, dropout), 3)

    def forward(self, x, memory, tgt_mask):
        m = memory # TODO：拼接memory和关键字信息,memory.size
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x,
                                                         tgt_mask))  # tgt_mask只在最开始的attention起作用（后续的隐向量均没有“当前token之后的信息“）   ??2,3,4....层还有必要遮盖??
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m))
        # q:x,k:m,v:m，输入和memory计算关联度，关联度*memory[i]得到output
        return self.sublayer[2](x, self.fnn)


def attention(query, key, value, mask=None, dropout=None):
    d_k = query.size(-1)  # d_k = d_model / n_head，单个head关注的维度
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    # 注意只转置两个维度，因此不能直接k.T
    # (32,8,seq_len,64)*(32,8,64,seq_len)-->(seq,8,seq_len,seq_len)。这里的值由64维信息得到；如果是非multi_head，就是由512维信息得到
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)  # -1*10^9
    p_attn = F.softmax(scores, dim=-1)
    if (dropout is not None):
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn
    # (1*dropout)


class MultiHeadAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        assert d_model % h == 0
        self.d_k = d_model // h
        self.h = h
        self.linear = _get_clones(nn.Linear(d_model,d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        if mask is not None:
            mask = mask.unsqueeze(
                1)  # src_mask:(32,1,seq_len)->(32,1,1,seq_len),前面的1在head上广播，后面的(1,seq_len)在(seq_len,seq_len)上广播
        n_batch = query.size(0)

        query, key, value = [l(x).view(n_batch, -1, self.h, self.d_k).transpose(1, 2) for l, x in
                             zip(self.linear, (query, key, value))]
        # (32,seq_len,512)-->(32, 8, seq_len, 64)
        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)
        # x:(n_batch,8,seq_len,64) attn:(n_batch, 8, seq_len, seq_len)
        x = x.transpose(1, 2).contiguous().view(n_batch, -1, self.h * self.d_k)
        # (32, 8, seq_len, 64)-->(32,seq_len,512)
        return self.linear[-1](x)


class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionWiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


class Generator(nn.Module):  # 和embedding正好相反，(n_batch,seq_len,d_model)-->(n_batch,seq_len,vocab),最后一维是概率分布
    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab)

    def forward(self, x):
        return F.log_softmax(self.proj(x), dim=-1)


def make_model(args):
    # src_vocab, tgt_vocab, N=6, d_model=512, d_ff=2048, h=8, dropout=0.1
    c = copy.deepcopy
    # 复用结构，不复用参数
    attn = MultiHeadAttention(args.n_heads, args.d_model)                   # (h, d_model, dropout=0.1)
    ffn = PositionWiseFeedForward(args.d_model, args.d_feedforward, args.dropout)   # (d_model, d_ff, dropout=0.1)
    position = PositionalEncoding(args.d_model, args.dropout)         # (emb_size: int, dropout, maxlen: int = 5000)
    if args.common_vocab:
        embedding1 = TokenEmbedding(args.src_vocab, args.d_model)
        embedding2 = embedding1
    else:
        embedding1 = TokenEmbedding(args.src_vocab, args.d_model)
        embedding2 = TokenEmbedding(args.tgt_vocab, args.d_model)
    model = EncoderDecoder(                                 # (encoder, decoder, src_embed, tgt_embed, generator)
        Encoder(EncoderLayer(args.d_model, c(attn), c(ffn), args.dropout), args.n_encoder_layers),
        # (encoder_layer, N)    # (size, self_attn, fnn, dropout)
        Decoder(DecoderLayer(args.d_model, c(attn), c(attn), c(ffn), args.dropout), args.n_decoder_layers),
        # (decoder_layer, N)    # (size, self_atten, src_atten, fnn, dropout)
        nn.Sequential(embedding1, c(position)),  # (vocab_size, d_model)
        nn.Sequential(embedding2, c(position)),  # 这两个网络的输入都不复杂，初始化后只接受一个tensor的输入
        Generator(args.d_model, args.tgt_vocab)                       # (d_model, vocab)
    )

    for p in model.parameters():
        if p.dim()>1:
            nn.init.xavier_uniform_(p)   # 初始化参数，使得每一层的方差尽可能相等。
    return model