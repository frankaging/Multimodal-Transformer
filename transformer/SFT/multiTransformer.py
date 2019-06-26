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

class MFN(nn.Module):
    def __init__(self, mods, dims, output_dim,
                 device=torch.device('cuda:0')):
        super(MFN, self).__init__()

        # input dims
        self.mods = mods
        self.dims = dims
        total_embed_size = 0
        total_hidden_size = 0
        self.hidden_dim = {'linguistic' : 88, 'emotient' : 16, 'acoustic' : 48, 'image' : 88}
        for mod in mods:
            total_embed_size += dims[mod]
            total_hidden_size += self.hidden_dim[mod]
        # config params TODO: from orginal paper https://github.com/pliang279/MFN/blob/master/test_mosi.py
        self.mem_dim = 128
        window_dim = 2
        attInShape = total_hidden_size*window_dim
        gammaInShape = attInShape+self.mem_dim
        final_out = total_hidden_size+self.mem_dim
        h_att1 = 128
        h_att2 = 256
        h_gamma1 = 64
        h_gamma2 = 64
        h_out = 64
        att1_dropout = 0.0
        att2_dropout = 0.0
        gamma1_dropout = 0.2
        gamma2_dropout = 0.2
        out_dropout = 0.5

        # lstm layers
        self.lstm = dict()
        for mod in mods:
            self.lstm[mod] = nn.LSTMCell(dims[mod], self.hidden_dim[mod])
            self.add_module('lstm_{}'.format(mod), self.lstm[mod])

        # layers
        self.att1_fc1 = nn.Linear(attInShape, h_att1)
        self.att1_fc2 = nn.Linear(h_att1, attInShape)
        self.att1_dropout = nn.Dropout(att1_dropout)

        self.att2_fc1 = nn.Linear(attInShape, h_att2)
        self.att2_fc2 = nn.Linear(h_att2, self.mem_dim)
        self.att2_dropout = nn.Dropout(att2_dropout)

        self.gamma1_fc1 = nn.Linear(gammaInShape, h_gamma1)
        self.gamma1_fc2 = nn.Linear(h_gamma1, self.mem_dim)
        self.gamma1_dropout = nn.Dropout(gamma1_dropout)

        self.gamma2_fc1 = nn.Linear(gammaInShape, h_gamma2)
        self.gamma2_fc2 = nn.Linear(h_gamma2, self.mem_dim)
        self.gamma2_dropout = nn.Dropout(gamma2_dropout)

        self.out_fc1 = nn.Linear(final_out, h_out)
        self.out_fc2 = nn.Linear(h_out, output_dim)
        self.out_dropout = nn.Dropout(out_dropout)

        # Store module in specified device (CUDA/CPU)
        self.device = (device if torch.cuda.is_available() else
                       torch.device('cpu'))
        self.to(self.device)

    def forward(self, inputs):
        # each input is t x n x d
        n = -1
        t = -1
        # construct needs for each mods
        # TODO: this assume cuda is avaliable?
        self.h = dict()
        self.c = dict()
        all_hs = dict()
        all_cs = dict()
        all_mems = []
        for mod in self.mods:
            input_mod = inputs[mod]
            n = input_mod.size()[1]
            t = input_mod.size()[0]
            self.h[mod] = torch.zeros(n, self.hidden_dim[mod]).to(self.device)
            self.c[mod] = torch.zeros(n, self.hidden_dim[mod]).to(self.device)
        self.mem = torch.zeros(n, self.mem_dim).to(self.device)

        for i in range(t):
            # prev time step
            prev_c = dict()
            for mod in self.mods:
                prev_c[mod] = self.c[mod]
            new_h = dict()
            new_c = dict()
            for mod in self.mods:
                new_h[mod], new_c[mod] = self.lstm[mod](inputs[mod][i], (self.h[mod], self.c[mod]))
            # concatenate
            prev_cs = []
            new_cs = []
            for mod in self.mods:
                prev_cs.append(prev_c[mod])
                new_cs.append(new_c[mod])
            prev_cs = torch.cat(prev_cs, dim=1)
            new_cs = torch.cat(new_cs, dim=1)
            cStar = torch.cat([prev_cs,new_cs], dim=1)
            attention = F.softmax(self.att1_fc2(self.att1_dropout(F.relu(self.att1_fc1(cStar)))),dim=1)
            attended = attention*cStar
            cHat = F.tanh(self.att2_fc2(self.att2_dropout(F.relu(self.att2_fc1(attended)))))
            both = torch.cat([attended,self.mem], dim=1)
            gamma1 = F.sigmoid(self.gamma1_fc2(self.gamma1_dropout(F.relu(self.gamma1_fc1(both)))))
            gamma2 = F.sigmoid(self.gamma2_fc2(self.gamma2_dropout(F.relu(self.gamma2_fc1(both)))))
            self.mem = gamma1*self.mem + gamma2*cHat
            all_mems.append(self.mem)
            # update
            for mod in self.mods:
                self.h[mod] = new_h[mod]
                self.c[mod] = new_c[mod]
                if mod not in all_hs.keys():
                    all_hs[mod] = []
                if mod not in all_cs.keys():
                    all_cs[mod] = []
                all_hs[mod].append(self.h[mod])
                all_cs[mod].append(self.c[mod])

        # combining to get the output at each time step
        outputs = []
        for i in range(t):
            last_hs = []
            for mod in self.mods:
                last_hs.append(all_hs[mod][i])
            last_hs.append(all_mems[i])
            last_hs = torch.cat(last_hs, dim=1)
            output = self.out_fc2(self.out_dropout(F.relu(self.out_fc1(last_hs))))
            outputs.append(output.unsqueeze(1))
        outputs = torch.cat(outputs, dim=1)
        return outputs

class MultiTransformer(nn.Module):
    def __init__(self, mods, window_embed_size,
                 N=6, d_ff=128, h=8, dropout=0.1, n_layers=1,
                 device=torch.device('cuda:0')):
        super(MultiTransformer, self).__init__()

        # modalities info and input dimensions
        self.mods = mods
        self.window_embed_size = window_embed_size
        # transformer embed layers
        self.embed_dim = {'linguistic' : 256, 'emotient' : 16, 'acoustic' : 256, 'image' : 256}

        self.embed = dict()
        self.transformer = dict()
        self.lstm = dict()
        self.attn = dict()
        self.ff = dict()
        c = copy.deepcopy
        for mod in mods:
            # for evert modality, we will have embed
            self.embed[mod] = nn.Linear(window_embed_size[mod], self.embed_dim[mod])
            self.add_module('embed_{}'.format(mod), self.embed[mod])
            # for evert modality, we will have a transformer
            self.attn[mod] = MultiHeadedAttention(h, self.embed_dim[mod])
            self.ff[mod] = PositionwiseFeedForward(self.embed_dim[mod], d_ff, dropout)
            self.add_module('attn{}'.format(mod), self.attn[mod])
            self.add_module('ff{}'.format(mod), self.ff[mod])
            self.transformer[mod] = Encoder(EncoderLayer(self.embed_dim[mod], c(self.attn[mod]), c(self.ff[mod]), dropout), N)
            self.add_module('transformer_{}'.format(mod), self.transformer[mod])

        # Memory fusion network to decode the outputs <- output dim = 1 TODO: check here!
        self.mfn = MFN(mods, self.embed_dim, 1)

        # Store module in specified device (CUDA/CPU)
        self.device = (device if torch.cuda.is_available() else
                       torch.device('cpu'))
        self.to(self.device)

    def forward(self, inputs, mask, lengths, tgt_init=0.5, target=None):
        # Get batch dim
        batch_size, seq_len = len(lengths), max(lengths)
        # Convert raw features into equal-dimensional embeddings
        mfn_in = dict()
        for mod in self.mods:
            # TODO: only linguistic cues will go through transformer
            #       otherwise just a linear layer
            embed = self.embed[mod](inputs[mod])
            # print("=== mod:" + mod + " ===")
            # print(inputs[mod])
            embed = self.transformer[mod](embed, mask) # batch_size, seq_len, self.embed_dim
            mfn_in[mod] = embed.permute(1,0,2) # seq_len, batch_size, self.embed_dim
            # print("=== mod:" + mod + " ===")
            # print(mfn_in[mod])
        predicted = self.mfn(mfn_in)
        # predicted = predicted.permute(1,0)
        # print("==mfn out size==")
        # print(predicted.size())
        # print("==mask size==")
        # print(mask.float().size())
        # Mask target entries that exceed sequence lengths
        predicted = predicted * mask.float()
        # print("==predicted size==")
        # print(predicted)
        return predicted

class UniTransformer(nn.Module):
    def __init__(self, window_embed_size, embed_dim=256, h_dim=128,
                 N=6, d_ff=128, h=8, dropout=0.1, n_layers=1,
                 device=torch.device('cuda:0')):
        super(UniTransformer, self).__init__()
        self.embed_dim = embed_dim
        self.h_dim = h_dim
        # embedding layers
        # Create raw-to-embed FC+Dropout layer
        self.embed = nn.Linear(window_embed_size, embed_dim)
        # modality -> only linguistics -> output embed_dim

        # encoder (6 encoders)
        # encoder = encoder layer + sublayer connection
        # encoder layer = attention layer + feedforward + norm layer
        c = copy.deepcopy
        attn = MultiHeadedAttention(h, embed_dim)
        ff = PositionwiseFeedForward(embed_dim, d_ff, dropout)
        self.encoder = Encoder(EncoderLayer(embed_dim, c(attn), c(ff), dropout), N)
        # Decodes targets and LSTM hidden states
        self.decoder = nn.LSTM(2*embed_dim, embed_dim, n_layers, batch_first=True)
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
        # p = torch.ones(batch_size, 1).to(self.device) * tgt_init
        o_prev = torch.zeros(batch_size, self.embed_dim).to(self.device)
        h, c = h0, c0
        for t in range(seq_len):
            # Concatenate prediction from previous timestep to context
            i = torch.cat([o_prev, encoder_output[:,t,:]], dim=1).unsqueeze(1)
            # Get next decoder LSTM state and output
            o, (h, c) = self.decoder(i, (h, c))
            o_prev = o.squeeze(1)
            # Computer prediction from output state
            p = self.out(o.view(-1, self.embed_dim))
            predicted.append(p.unsqueeze(1))
            # print(predicted)
        predicted = torch.cat(predicted, dim=1)
        # Mask target entries that exceed sequence lengths
        predicted = predicted * mask.float()
        return predicted

class UniFullTransformer(nn.Module):
    def __init__(self, window_embed_size, embed_dim=256, h_dim=128,
                 N=6, d_ff=128, h=8, dropout=0.1, n_layers=1,
                 device=torch.device('cuda:0')):
        super(UniFullTransformer, self).__init__()
        self.embed_dim = embed_dim
        self.h_dim = h_dim
        # embedding layers
        # Create raw-to-embed FC+Dropout layer
        self.embed = nn.Linear(window_embed_size, embed_dim)
        # modality -> only linguistics -> output embed_dim

        # encoder (6 encoders)
        # encoder = encoder layer + sublayer connection
        # encoder layer = attention layer + feedforward + norm layer
        c = copy.deepcopy
        attn = MultiHeadedAttention(h, embed_dim)
        ff = PositionwiseFeedForward(embed_dim, d_ff, dropout)
        self.encoder = Encoder(EncoderLayer(embed_dim, c(attn), c(ff), dropout), N)

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
        # print(encoder_output.size())
        predicted = self.out(encoder_output) # <- embed to 1
        # print(predicted)
        # Mask target entries that exceed sequence lengths
        predicted = predicted * mask.float()
        return predicted

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
        self.decoder = nn.LSTM(2*embed_dim, embed_dim, n_layers, batch_first=True)
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
        # p = torch.ones(batch_size, 1).to(self.device) * tgt_init
        o_prev = torch.zeros(batch_size, self.embed_dim).to(self.device)
        h, c = h0, c0
        for t in range(seq_len):
            # Concatenate prediction from previous timestep to context
            i = torch.cat([o_prev, encoder_output[:,t,:]], dim=1).unsqueeze(1)
            # Get next decoder LSTM state and output
            o, (h, c) = self.decoder(i, (h, c))
            o_prev = o.squeeze(1)
            # Computer prediction from output state
            p = self.out(o.view(-1, self.embed_dim))
            predicted.append(p.unsqueeze(1))
            # print(predicted)
        predicted = torch.cat(predicted, dim=1)
        # Mask target entries that exceed sequence lengths
        predicted = predicted * mask.float()
        return predicted
