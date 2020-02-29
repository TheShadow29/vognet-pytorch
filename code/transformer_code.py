"""
Transformer implementation adapted from
https://github.com/facebookresearch/grounded-video-description/blob/master/misc/transformer.py
"""
import torch
import math
from torch import nn
from torch.nn import functional as F

INF = 1e10


def matmul(x, y):
    if x.dim() == y.dim():
        return torch.matmul(x, y)
    if x.dim() == y.dim() - 1:
        return torch.matmul(x.unsqueeze(-2), y).squeeze(-2)
    return torch.matmul(x, y.unsqueeze(-2)).squeeze(-2)


class ResidualBlock(nn.Module):

    def __init__(self, layer, d_model, drop_ratio):
        super(ResidualBlock, self).__init__()
        self.layer = layer
        self.dropout = nn.Dropout(drop_ratio)
        # self.layernorm = LayerNorm(d_model)
        self.layernorm = nn.LayerNorm(d_model)

    def forward(self, *x):
        return self.layernorm(x[0] + self.dropout(self.layer(*x)))


class Attention(nn.Module):

    def __init__(self, d_key, drop_ratio, causal):
        super(Attention, self).__init__()
        self.scale = math.sqrt(d_key)
        self.dropout = nn.Dropout(drop_ratio)
        self.causal = causal

    def forward(self, query, key, value):
        dot_products = matmul(query, key.transpose(1, 2))
        if query.dim() == 3 and (self is None or self.causal):
            tri = torch.ones(key.size(1), key.size(1)).triu(1) * INF
            if key.is_cuda:
                tri = tri.cuda(key.get_device())
            dot_products.data.sub_(tri.unsqueeze(0))

        return matmul(self.dropout(F.softmax(dot_products / self.scale, dim=-1)), value)


class MultiHead(nn.Module):

    def __init__(self, d_key, d_value, n_heads, drop_ratio, causal=False):
        super(MultiHead, self).__init__()
        self.attention = Attention(d_key, drop_ratio, causal=causal)
        self.wq = nn.Linear(d_key, d_key, bias=False)
        self.wk = nn.Linear(d_key, d_key, bias=False)
        self.wv = nn.Linear(d_value, d_value, bias=False)
        self.wo = nn.Linear(d_value, d_key, bias=False)
        self.n_heads = n_heads

    def forward(self, query, key, value):
        query, key, value = self.wq(query), self.wk(key), self.wv(value)

        query, key, value = (
            x.chunk(self.n_heads, -1) for x in (query, key, value))
        return self.wo(torch.cat([self.attention(q, k, v)
                                  for q, k, v in zip(query, key, value)], -1))


class FeedForward(nn.Module):

    def __init__(self, d_model, d_hidden):
        super(FeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, d_hidden)
        self.linear2 = nn.Linear(d_hidden, d_model)

    def forward(self, x):
        return self.linear2(F.relu(self.linear1(x)))


class EncoderLayer(nn.Module):

    def __init__(self, d_model, d_hidden, n_heads, drop_ratio):
        super(EncoderLayer, self).__init__()
        self.selfattn = ResidualBlock(
            MultiHead(d_model, d_model, n_heads, drop_ratio),
            d_model, drop_ratio)
        self.feedforward = ResidualBlock(FeedForward(d_model, d_hidden),
                                         d_model, drop_ratio)

    def forward(self, x):
        return self.feedforward(self.selfattn(x, x, x))


class Encoder(nn.Module):

    def __init__(self, d_model, d_hidden, n_vocab, n_layers, n_heads,
                 drop_ratio, pe):
        super(Encoder, self).__init__()
        # self.linear = nn.Linear(d_model*2, d_model)
        self.layers = nn.ModuleList(
            [EncoderLayer(d_model, d_hidden, n_heads, drop_ratio)
             for i in range(n_layers)])
        self.dropout = nn.Dropout(drop_ratio)
        self.pe = pe

    def forward(self, x, mask=None):
        # x = self.linear(x)
        if self.pe:
            # spatial configuration is already encoded
            # x = x+positional_encodings_like(x)
            raise NotImplementedError
        # x = self.dropout(x) # dropout is already in the pool_embed layer
        if mask is not None:
            x = x*mask
        encoding = []
        for layer in self.layers:
            x = layer(x)
            if mask is not None:
                x = x*mask
            encoding.append(x)
        return encoding


class RelAttention(nn.Module):

    def __init__(self, d_key, drop_ratio, causal):
        super().__init__()
        self.scale = math.sqrt(d_key)
        self.dropout = nn.Dropout(drop_ratio)
        self.causal = causal

    def forward(self, query, key, value, pe_k, pe_v):
        """
        query, key, value: B x N x 214
        pe_k: B x N x N x 214
        """
        dot_products = matmul(query, key.transpose(1, 2))
        if query.dim() == 3 and (self is None or self.causal):
            tri = torch.ones(key.size(1), key.size(1)).triu(1) * INF
            if key.is_cuda:
                tri = tri.cuda(key.get_device())
            dot_products.data.sub_(tri.unsqueeze(0))

        # new_dp = matmul(query, pe_k.transpose(2, 3))
        new_dp = pe_k.squeeze(-1)
        assert new_dp.shape == dot_products.shape
        new_dot_prods = (dot_products + new_dp) / self.scale

        attn = self.dropout(F.softmax(new_dot_prods, dim=-1))

        out_v = matmul(attn, value)
        # new_out_v = matmul(attn, pe_v)
        # new_out_v = pe_v

        new_outs = out_v
        return new_outs


class RelMultiHead(nn.Module):

    def __init__(self, d_key, d_value, n_heads, drop_ratio, causal=False, d_pe=None):
        super().__init__()
        self.attention = RelAttention(d_key, drop_ratio, causal=causal)
        self.n_heads = n_heads
        self.wq = nn.Linear(d_key, d_key, bias=False)
        self.wk = nn.Linear(d_key, d_key, bias=False)
        self.wv = nn.Linear(d_value, d_value, bias=False)
        self.wo = nn.Linear(d_value, d_key, bias=False)
        # self.wpk = nn.Linear(d_pe, self.n_heads, bias=False)
        # self.wpv = nn.Linear(d_pe, self.n_heads, bias=False)

    def forward(self, query, key, value, pe=None):
        """
        pe is B x N x N x 1 position difference
        """
        query, key, value = self.wq(query), self.wk(key), self.wv(value)
        pe_k, pe_v = pe, pe
        query, key, value, pe_k, pe_v = (
            x.chunk(self.n_heads, -1) for x in (query, key, value, pe_k, pe_v))
        return self.wo(torch.cat([self.attention(q, k, v, pk, pv)
                                  for q, k, v, pk, pv in
                                  zip(query, key, value, pe_k, pe_v)], -1))


class RelEncoderLayer(nn.Module):

    def __init__(self, d_model, d_hidden, n_heads,
                 drop_ratio, d_pe=None, sa=True):
        super().__init__()
        self.selfattn = ResidualBlock(
            RelMultiHead(d_model, d_model, n_heads, drop_ratio, d_pe=d_pe),
            d_model, drop_ratio)
        self.feedforward = ResidualBlock(FeedForward(d_model, d_hidden),
                                         d_model, drop_ratio)
        self.sa = sa

    def forward(self, x, pe=None):
        if not isinstance(x, dict):
            return self.feedforward(self.selfattn(x, x, x, pe))
        else:
            assert not self.sa
            assert isinstance(x, dict)
            assert 'query' in x
            assert 'key' in x
            assert 'value' in x
            return self.feedforward(
                self.selfattn(x['query'], x['key'], x['value'], pe)
            )


class RelEncoder(nn.Module):

    def __init__(self, d_model, d_hidden, n_vocab, n_layers, n_heads,
                 drop_ratio, pe, d_pe, sa=True):
        super().__init__()
        # self.linear = nn.Linear(d_model*2, d_model)
        self.layers = nn.ModuleList(
            [RelEncoderLayer(d_model, d_hidden, n_heads, drop_ratio, d_pe=d_pe, sa=sa)
             for i in range(n_layers)])
        self.dropout = nn.Dropout(drop_ratio)
        self.pe = pe

    def forward(self, x, x_pe, mask=None):
        # x = self.linear(x)
        if self.pe:
            # spatial configuration is already encoded
            raise NotImplementedError
        # x = self.dropout(x) # dropout is already in the pool_embed layer
        if mask is not None:
            x = x*mask
        encoding = []
        for layer in self.layers:
            x = layer(x, pe=x_pe)
            if mask is not None:
                x = x*mask
            encoding.append(x)
        return encoding


class Transformer(nn.Module):

    def __init__(self, d_model, n_vocab_src, vocab_trg, d_hidden=2048,
                 n_layers=6, n_heads=8, drop_ratio=0.1, pe=False):
        super(Transformer, self).__init__()
        self.encoder = Encoder(d_model, d_hidden, n_vocab_src, n_layers,
                               n_heads, drop_ratio, pe)

    def forward(self, x):
        encoding = self.encoder(x)
        return encoding[-1]
        # return encoding[-1], encoding
        # return torch.cat(encoding, 2)

    def all_outputs(self, x):
        encoding = self.encoder(x)
        return encoding


class RelTransformer(nn.Module):

    def __init__(self, d_model, n_vocab_src, vocab_trg, d_hidden=2048,
                 n_layers=6, n_heads=8, drop_ratio=0.1, pe=False, d_pe=None):
        super().__init__()
        self.encoder = RelEncoder(d_model, d_hidden, n_vocab_src, n_layers,
                                  n_heads, drop_ratio, pe, d_pe=d_pe)

    def forward(self, x, x_pe):
        encoding = self.encoder(x, x_pe)
        return encoding[-1]
        # return encoding[-1], encoding
        # return torch.cat(encoding, 2)

    def all_outputs(self, x):
        encoding = self.encoder(x)
        return encoding
