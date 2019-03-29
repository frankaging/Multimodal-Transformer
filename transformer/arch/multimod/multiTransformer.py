import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math, copy, time
from torch.autograd import Variable
import matplotlib.pyplot as plt

class PositionwiseFeedForward(nn.Module):
    '''
    Feed-forward layer
    '''
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))

def attention(query, key, value, mask=None, dropout=None):
    '''
    Helper function for calculating attention
    '''
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) \
             / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim = -1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn

class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)
        
    def forward(self, query, key, value, mask=None):
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)
        
        # 1) Do all the linear projections in batch from d_model => h x d_k 
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]
        
        # 2) Apply attention on all the projected vectors in batch. 
        x, self.attn = attention(query, key, value, mask=mask, 
                                 dropout=self.dropout)
        
        # 3) "Concat" using a view and apply a final linear. 
        x = x.transpose(1, 2).contiguous() \
             .view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)

class Encoder(nn.Module):
    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)
        
    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)

def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

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

class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))

class EncoderLayer(nn.Module):
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)

class NLPTransformer(nn.Module):
    def __init__(self, window_embed_size, embed_dim=256, h_dim=128, 
                 N=6, d_ff=128, h=8, dropout=0.1, n_layers=1,
                 device=torch.device('cuda:0')):
        super(NLPTransformer, self).__init__()
        self.embed_dim = embed_dim
        self.h_dim = h_dim
        # embedding layers
        # Create raw-to-embed FC+Dropout layer
        self.embed = nn.Sequential(nn.Dropout(0.1),
                                   nn.Linear(window_embed_size, embed_dim),
                                   nn.ReLU())
        # modality -> only linguistics -> output embed_dim

        # encoder (6 encoders)
        # encoder = encoder layer + sublayer connection
        # encoder layer = attention layer + feedforward + norm layer
        c = copy.deepcopy
        attn = MultiHeadedAttention(h, embed_dim)
        ff = PositionwiseFeedForward(embed_dim, d_ff, dropout)
        self.encoder = Encoder(EncoderLayer(embed_dim, c(attn), c(ff), dropout), N)
        # Decodes targets and LSTM hidden states
        self.decoder = nn.LSTM(1+embed_dim, embed_dim, n_layers, batch_first=True)
        self.dec_h0 = nn.Parameter(torch.zeros(n_layers, 1, embed_dim))
        self.dec_c0 = nn.Parameter(torch.zeros(n_layers, 1, embed_dim))
        # the output will be in the embed_dim dimension
        # output only 1d
        self.out = nn.Sequential(nn.Linear(embed_dim, h_dim),
                                 nn.ReLU(),
                                 nn.Linear(h_dim, 1))
        # Store module in specified device (CUDA/CPU)
        self.device = (device if torch.cuda.is_available() else
                       torch.device('cpu'))
        self.to(self.device)

    def forward(self, inputs, mask, lengths, tgt_init=0.5, target=None):
        # Get batch dim
        batch_size, seq_len = len(lengths), max(lengths)
        # Convert raw features into equal-dimensional embeddings
        embed = self.embed(inputs)
        encoder_output = self.encoder(embed, mask) # batch_size, seq_len, self.embed_dim
        # LSTM output from the encoder
        # Set initial hidden and cell states for decoder
        h0 = self.dec_h0.repeat(1, batch_size, 1)
        c0 = self.dec_c0.repeat(1, batch_size, 1)
        predicted = []
        p = torch.ones(batch_size, 1).to(self.device) * tgt_init
        # o_prev = torch.zeros(batch_size, self.embed_dim).to(self.device)
        h, c = h0, c0
        for t in range(seq_len):
            # Concatenate prediction from previous timestep to context
            i = torch.cat([p, encoder_output[:,t,:]], dim=1).unsqueeze(1)
            # Get next decoder LSTM state and output
            o, (h, c) = self.decoder(i, (h, c))
            # o_prev = o.squeeze(1)
            # Computer prediction from output state
            p = self.out(o.view(-1, self.embed_dim))
            predicted.append(p.unsqueeze(1))
            # print(predicted)
        predicted = torch.cat(predicted, dim=1)
        # Mask target entries that exceed sequence lengths
        predicted = predicted * mask.float()
        return predicted
