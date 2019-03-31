from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from multiTransformer import NLPTransformer

def pad_shift(x, shift, padv=0.0):
    """Shift 3D tensor forwards in time with padding."""
    if shift > 0:
        padding = torch.ones(x.size(0), shift, x.size(2)).to(x.device) * padv
        return torch.cat((padding, x[:, :-shift, :]), dim=1)
    elif shift < 0:
        padding = torch.ones(x.size(0), -shift, x.size(2)).to(x.device) * padv
        return torch.cat((x[:, -shift:, :], padding), dim=1)
    else:
        return x

def convolve(x, attn):
    """Convolve 3D tensor (x) with local attention weights (attn)."""
    stacked = torch.stack([pad_shift(x, i) for
                           i in range(attn.shape[2])], dim=-1)
    return torch.sum(attn.unsqueeze(2) * stacked, dim=-1)

class VGG16(nn.Module):
    ''' input = (1, 1, 224, 224) '''
    def __init__(self, window_embed_size=128, num_classes=256,
                 dropout=0.1, device=torch.device('cuda:0')):
        super(VGG16, self).__init__()
        self.num_classes = num_classes
        self.convs = nn.Sequential(
                        nn.Conv2d(1, 64, kernel_size=3, padding=1),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(64, 64, kernel_size=3, padding=1),
                        nn.ReLU(inplace=True),
                        nn.MaxPool2d(kernel_size=2, stride=2),
                        nn.Conv2d(64, 128, kernel_size=3, padding=1),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(128, 128, kernel_size=3, padding=1),
                        nn.ReLU(inplace=True),
                        nn.MaxPool2d(kernel_size=2, stride=2),
                        nn.Conv2d(128, 256, kernel_size=3, padding=1),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(256, 256, kernel_size=3, padding=1),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(256, 256, kernel_size=3, padding=1),
                        nn.ReLU(inplace=True),
                        nn.MaxPool2d(kernel_size=2, stride=2),
                        nn.Conv2d(256, 512, kernel_size=3, padding=1),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(512, 512, kernel_size=3, padding=1),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(512, 512, kernel_size=3, padding=1),
                        nn.ReLU(inplace=True),
                        nn.MaxPool2d(kernel_size=2, stride=2),
                        nn.Conv2d(512, 512, kernel_size=3, padding=1),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(512, 512, kernel_size=3, padding=1),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(512, 512, kernel_size=3, padding=1),
                        nn.ReLU(inplace=True),
                        nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Sequential(
                            nn.Linear(512 * 3 * 3, 4096),
                            nn.ReLU(inplace=True),
                            nn.Dropout(),
                            nn.Linear(4096, 4096),
                            nn.ReLU(inplace=True),
                            nn.Dropout(),
                            nn.Linear(4096, num_classes),
        )

        # Store module in specified device (CUDA/CPU)
        self.device = (device if torch.cuda.is_available() else
                       torch.device('cpu'))
        self.to(self.device)

    def forward(self, inputs):
        x = self.convs(inputs)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

class Highway(nn.Module):
	def __init__(self, word_embed_size):
		"""
        Init the Highway
        @param word_embed_size (int): Embedding size (dimensionality) for the output
        """
		super(Highway, self).__init__()
		# TODO:
		# 1. construct a linear layer with bias
		# 2. another linear layer with bias
		self.word_embed_size = word_embed_size
		self.linear_projection = nn.Linear(self.word_embed_size, self.word_embed_size, bias=True)
		self.linear_gate = nn.Linear(self.word_embed_size, self.word_embed_size, bias=True)

	def forward(self, x_conv_out: torch.Tensor) -> torch.Tensor:
		"""
		Highway forward function
		@param x_conv_out: (batch_size, char_embed_size)
		"""
		# TODO:
		# input: x_reshape is a tensor in dimension (batch_size, word_embed_size)
		# 1. apply ReLU
		# 2. apply sigma
		# 3. apply highway function
		# *. consider this is in a batch operation. get the input length for example
		x_proj = nn.functional.relu(self.linear_projection(x_conv_out))
		x_gate = nn.functional.sigmoid(self.linear_gate(x_conv_out))
		x_highway = (x_gate * x_proj) + ((1 - x_gate) * x_conv_out)
		return x_highway

class CNN(nn.Module):
    def __init__(self, word_embed_size=300, window_embed_size=128, k=3):
        super(CNN, self).__init__()
        # TODO:
        # 1. con1d layer, with input and output size
        self.k = k
        self.f = window_embed_size
        self.word_embed_size = word_embed_size
        self.window_embed_size = window_embed_size
        self.conv1d = nn.Conv1d(self.word_embed_size, self.f, self.k, bias=True)

    def forward(self, x_reshape: torch.Tensor) -> torch.Tensor:
        # TODO:
        # input: input is a tensor in shape (batch_size, word_embed_size, max_window_length)
        # output: output is a tensor in shape (batch_size, window_embed_size)
        x_conv = self.conv1d(x_reshape) # (batch_size, window_embed_size, m_word+k+1)
        # x_conv_relu = nn.functional.relu(x_conv)
        L = x_conv.size()[2]
        maxpool = nn.MaxPool1d(L, stride=3)
        x_conv_out = torch.squeeze(maxpool(x_conv), 2) # (batch_size, window_embed_size)
        return x_conv_out

class MultiCNNLSTM(nn.Module):
    def __init__(self, mods, dims, fuse_embed_size=256, k=3,
                 device=torch.device('cuda:0')):
        super(MultiCNNLSTM, self).__init__()
        # init
        self.mods = mods
        self.dims = dims
        self.CNN = dict()
        self.Highway = dict()
        self.window_embed_size={'linguistic' : 256, 'emotient' : 128, 'acoustic' : 256, 'image' : 256}
        total_embed_size = 0
        for mod in mods:
            if mod != 'image':
                self.CNN[mod] = CNN(dims[mod], self.window_embed_size[mod], k)
                self.Highway[mod] = Highway(self.window_embed_size[mod])
                self.add_module('cnn_{}'.format(mod), self.CNN[mod])
                self.add_module('highway_{}'.format(mod), self.Highway[mod])
            total_embed_size += self.window_embed_size[mod]
        self.VGG = VGG16()
        self.fusionLayer = nn.Linear(total_embed_size, fuse_embed_size)
        if len(mods) > 1:
            self.LSTM = NLPTransformer(fuse_embed_size)
        else:
            self.LSTM = NLPTransformer(total_embed_size)
        self.dropout = nn.Dropout(p=0.3)
        # Store module in specified device (CUDA/CPU)
        self.device = (device if torch.cuda.is_available() else
                       torch.device('cpu'))
        self.to(self.device)

    def forward(self, inputs, length, mask=None):
        '''
        inputs = (batch_size, 39, 33, 300)
        '''
        outputs = []
        for mod in self.mods:
            inputs_mod = inputs[mod]
            outputs_mod = []
            if mod == 'image':
                for x in torch.split(inputs_mod, 1, 0):  # input -> (batch_size, 39, 33, 10000)
                    # print(x.size())
                    x = torch.squeeze(x, 0) # input -> (39, ~max window, 10000)
                    # print(x.size())
                    x = x.reshape(-1, 1, 100, 100)
                    vggout = self.VGG(x) # -> (39, 256)
                    # print(vggout.size())
                    outputs_mod.append(vggout)
                outputs_mod = torch.stack(outputs_mod, dim=0) # -> (batch_size, seq_l, 256)
            else:
                for x in torch.split(inputs_mod, 1, 0):
                    print('current memory allocated: {}'.format(torch.cuda.memory_allocated() / 1024 ** 2))
                    print('max memory allocated: {}'.format(torch.cuda.max_memory_allocated() / 1024 ** 2))
                    print('cached memory: {}'.format(torch.cuda.memory_cached() / 1024 ** 2))
                    x = torch.squeeze(x, 0) # input -> (39, 33, 300)
                    cnnOut = self.CNN[mod](x.permute(0, 2, 1)) # -> (39, 128)
                    x_highway = self.Highway[mod](cnnOut)
                    x_word_emb = self.dropout(x_highway)
                    outputs_mod.append(x_word_emb)
                outputs_mod = torch.stack(outputs_mod, dim=0)

            outputs.append(outputs_mod)
        if len(outputs) > 1:
            outputs = torch.cat(outputs, 2)
            fused_outputs = torch.tanh(self.fusionLayer(outputs))
        else:
            fused_outputs = outputs[0]
        predict = self.LSTM(fused_outputs, mask, length)
        return predict

class MultiLSTM(nn.Module):
    """Multimodal LSTM model with feature level fusion.

    modalities -- list of names of each input modality
    dims -- list of dimensions for input modalities
    embed_dim -- dimensions of embedding for feature-level fusion
    h_dim -- dimensions of LSTM hidden state
    n_layers -- number of LSTM layers
    attn_len -- length of local attention window
    """

    def __init__(self, window_embed_size, embed_dim=128, h_dim=256,
                 n_layers=1, attn_len=5, device=torch.device('cuda:0')):
        super(MultiLSTM, self).__init__()

        self.embed_dim = embed_dim
        self.h_dim = h_dim
        self.n_layers = n_layers
        self.attn_len = attn_len

        # Create raw-to-embed FC+Dropout layer
        self.embed = nn.Sequential(nn.Dropout(0.1),
                                   nn.Linear(window_embed_size, embed_dim),
                                   nn.ReLU())

        # Layer that computes attention from embeddings
        self.attn = nn.Sequential(nn.Linear(embed_dim, embed_dim),
                                  nn.ReLU(),
                                  nn.Linear(embed_dim, attn_len),
                                  nn.Softmax(dim=1))
        # LSTM computes hidden states from embeddings for each modality
        self.lstm = nn.LSTM(embed_dim, h_dim,
                            n_layers, batch_first=True)
        # Regression network from LSTM hidden states to predicted valence
        self.decoder = nn.Sequential(nn.Linear(h_dim, embed_dim),
                                     nn.ReLU(),
                                     nn.Linear(embed_dim, 1))
        # Store module in specified device (CUDA/CPU)
        self.device = (device if torch.cuda.is_available() else
                       torch.device('cpu'))
        self.to(self.device)

    def forward(self, inputs, mask, lengths, target=None, output_feats=False):
        # Get batch dim
        # print(lengths)
        # print(inputs.size())
        # print(mask.size())
        batch_size, seq_len = len(lengths), max(lengths)
        # Convert raw features into equal-dimensional embeddings
        embed = self.embed(inputs)
        # Compute attention weights
        attn = self.attn(embed)
        # Unflatten temporal dimension
        embed = embed.reshape(batch_size, seq_len, self.embed_dim)
        attn = attn.reshape(batch_size, seq_len, self.attn_len)
        # Pack the input to mask padded entries
        embed = pack_padded_sequence(embed, lengths, batch_first=True)
        # Set initial hidden and cell states
        h0 = torch.zeros(self.n_layers, batch_size, self.h_dim).to(self.device)
        c0 = torch.zeros(self.n_layers, batch_size, self.h_dim).to(self.device)
        # Forward propagate LSTM
        h, _ = self.lstm(embed, (h0, c0))
        # Undo the packing
        h, _ = pad_packed_sequence(h, batch_first=True)
        # Convolve output with attention weights
        # i.e. out[t] = a[t,0]*in[t] + ... + a[t,win_len-1]*in[t-(win_len-1)]
        context = convolve(h, attn)
        # Flatten temporal dimension
        context = context.reshape(-1, self.h_dim)
        # Return features before final FC layer if flag is set
        # if output_feats:
        #     features = self.decoder[0](context)
        #     features = features.view(batch_size, seq_len, -1) * mask.float()
        #     return features
        # Decode the context for each time step
        target = self.decoder(context).view(batch_size, seq_len, 1)
        # Mask target entries that exceed sequence lengths
        # print(target.size())
        # print(mask.size())
        target = target * mask.float()
        return target

class MultiEDLSTM(nn.Module):
    """Multimodal encoder-decoder LSTM model.

    modalities -- list of names of each input modality
    dims -- list of dimensions for input modalities
    embed_dim -- dimensions of embedding for feature-level fusion
    h_dim -- dimensions of LSTM hidden state
    n_layers -- number of LSTM layers
    attn_len -- length of local attention window
    """

    def __init__(self, window_embed_size, embed_dim=128, h_dim=512,
                 n_layers=1, attn_len=3, device=torch.device('cuda:0')):
        super(MultiEDLSTM, self).__init__()
        self.embed_dim = embed_dim
        self.h_dim = h_dim
        self.n_layers = n_layers
        self.attn_len = attn_len

        # Create raw-to-embed FC+Dropout layer
        self.embed = nn.Sequential(nn.Dropout(0.1),
                                   nn.Linear(window_embed_size, embed_dim),
                                   nn.ReLU())
        # Layer that computes attention from embeddings
        self.attn = nn.Sequential(nn.Linear(embed_dim, embed_dim),
                                  nn.ReLU(),
                                  nn.Linear(embed_dim, attn_len),
                                  nn.Softmax(dim=1))
        # Encoder computes hidden states from embeddings for each modality
        self.encoder = nn.LSTM(embed_dim, h_dim,
                               n_layers, batch_first=True)
        self.enc_h0 = nn.Parameter(torch.zeros(n_layers, 1, h_dim))
        self.enc_c0 = nn.Parameter(torch.zeros(n_layers, 1, h_dim))
        # Decodes targets and LSTM hidden states
        self.decoder = nn.LSTM(1 + h_dim, h_dim, n_layers, batch_first=True)
        self.dec_h0 = nn.Parameter(torch.zeros(n_layers, 1, h_dim))
        self.dec_c0 = nn.Parameter(torch.zeros(n_layers, 1, h_dim))
        # Final MLP output network
        self.out = nn.Sequential(nn.Linear(h_dim, embed_dim),
                                 nn.ReLU(),
                                 nn.Linear(embed_dim, 1))
        # Store module in specified device (CUDA/CPU)
        self.device = (device if torch.cuda.is_available() else
                       torch.device('cpu'))
        self.to(self.device)

    def forward(self, inputs, mask, lengths, target=None, tgt_init=0.0):
        # Get batch dim
        batch_size, seq_len = len(lengths), max(lengths)
        # Set initial hidden and cell states for encoder
        h0 = self.enc_h0.repeat(1, batch_size, 1)
        c0 = self.enc_c0.repeat(1, batch_size, 1)
        # Convert raw features into equal-dimensional embeddings
        embed = self.embed(inputs)
        # Compute attention weights
        attn = self.attn(embed)
        # Unflatten temporal dimension
        embed = embed.reshape(batch_size, seq_len, self.embed_dim)
        attn = attn.reshape(batch_size, seq_len, self.attn_len)
        # Pack the input to mask padded entries
        embed = pack_padded_sequence(embed, lengths, batch_first=True)
        # Forward propagate encoder LSTM
        enc_out, _ = self.encoder(embed, (h0, c0))
        # Undo the packing
        enc_out, _ = pad_packed_sequence(enc_out, batch_first=True)
        # Convolve output with attention weights
        # i.e. out[t] = a[t,0]*in[t] + ... + a[t,win_len-1]*in[t-(win_len-1)]
        context = convolve(enc_out, attn)
        # Set initial hidden and cell states for decoder
        h0 = self.dec_h0.repeat(1, batch_size, 1)
        c0 = self.dec_c0.repeat(1, batch_size, 1)
        # Use earlier predictions to predict next time-steps
        predicted = []
        p = torch.ones(batch_size, 1).to(self.device) * tgt_init
        h, c = h0, c0
        for t in range(seq_len):
            # Concatenate prediction from previous timestep to context
            i = torch.cat([p, context[:,t,:]], dim=1).unsqueeze(1)
            # Get next decoder LSTM state and output
            o, (h, c) = self.decoder(i, (h, c))
            # Computer prediction from output state
            p = self.out(o.view(-1, self.h_dim))
            predicted.append(p.unsqueeze(1))
        predicted = torch.cat(predicted, dim=1)
        # Mask target entries that exceed sequence lengths
        predicted = predicted * mask.float()
        return predicted

class MultiARLSTM(nn.Module):
    """Multimodal LSTM model with auto-regressive final layer.

    modalities -- list of names of each input modality
    dims -- list of dimensions for input modalities
    embed_dim -- dimensions of embedding for feature-level fusion
    h_dim -- dimensions of LSTM hidden state
    n_layers -- number of LSTM layers
    attn_len -- length of local attention window
    ar_order -- autoregressive order (i.e. length of AR window)
    """

    def __init__(self, window_embed_size, embed_dim=128, h_dim=512, n_layers=1,
                 attn_len=7, ar_order=1, device=torch.device('cuda:0')):
        super(MultiARLSTM, self).__init__()
        self.embed_dim = embed_dim
        self.h_dim = h_dim
        self.n_layers = n_layers
        self.attn_len = attn_len
        self.ar_order = ar_order

        # Create raw-to-embed FC+Dropout layer for each modality
        self.embed = nn.Sequential(nn.Dropout(0.1),
                                   nn.Linear(window_embed_size, embed_dim),
                                   nn.ReLU())
        # Computes attention from embeddings
        self.attn = nn.Sequential(nn.Linear(embed_dim, embed_dim),
                                  nn.ReLU(),
                                  nn.Linear(embed_dim, attn_len),
                                  nn.Softmax(dim=1))
        # LSTM computes hidden states from embeddings for each modality
        self.lstm = nn.LSTM(embed_dim, h_dim,
                            n_layers, batch_first=True)
        # Decodes LSTM hidden states into contribution to output term
        self.decoder = nn.Sequential(nn.Linear(h_dim, embed_dim),
                                     nn.ReLU(),
                                     nn.Linear(embed_dim, 1))
        # Computes autoregressive weight on previous output
        self.autoreg = nn.Linear(h_dim, ar_order)
        # Store module in specified device (CUDA/CPU)
        self.device = (device if torch.cuda.is_available() else
                       torch.device('cpu'))
        self.to(self.device)

    def forward(self, inputs, mask, lengths, target=None, tgt_init=0.0):
        # Get batch dim
        batch_size, seq_len = len(lengths), max(lengths)
        # Convert raw features into equal-dimensional embeddings
        embed = self.embed(inputs)
        # Compute attention weights
        attn = self.attn(embed)
        # Unflatten temporal dimension
        embed = embed.reshape(batch_size, seq_len, self.embed_dim)
        attn = attn.reshape(batch_size, seq_len, self.attn_len)
        # Pack the input to mask padded entries
        embed = pack_padded_sequence(embed, lengths, batch_first=True)
        # Forward propagate LSTM
        h, _ = self.lstm(embed)
        # Undo the packing
        h, _ = pad_packed_sequence(h, batch_first=True)
        # Convolve output with attention weights
        # i.e. out[t] = a[t,0]*in[t] + ... + a[t,win_len-1]*in[t-(win_len-1)]
        context = convolve(h, attn)
        # Flatten temporal dimension
        context = context.reshape(-1, self.h_dim)
        # Decode the context for each time step
        in_part = self.decoder(context).view(batch_size, seq_len, 1)
        # Compute autoregression weights
        ar_weight = self.autoreg(context)
        ar_weight = ar_weight.reshape(batch_size, seq_len, self.ar_order)
        # Compute predictions as autoregressive sum
        if target is not None:
            # Use teacher forcing
            ar_stacked = torch.stack([pad_shift(target, i) for
                                      i in range(self.ar_order)], dim=-1)
            ar_part = torch.sum(ar_weight.unsqueeze(2) * ar_stacked, dim=-1)
            predicted = in_part + ar_part
        else:
            # Otherwise use own predictions
            p_init = torch.ones(batch_size, 1).to(self.device) * tgt_init
            predicted = [p_init] * self.ar_order
            for t in range(seq_len):
                ar_hist = [p.detach() for p in predicted[-self.ar_order:]]
                ar_hist = torch.cat(ar_hist, dim=1)
                ar_part = torch.sum(ar_weight[:,t,:] * ar_hist, dim=1)
                p = in_part[:,t,:] + ar_part.unsqueeze(-1)
                predicted.append(p)
            predicted = torch.cat(predicted[self.ar_order:], 1).unsqueeze(-1)
        # Mask predicted entries that exceed sequence lengths
        predicted = predicted * mask.float()
        return predicted

if __name__ == "__main__":
    # Test code by loading dataset and running through model
    import os, argparse
    from datasets import load_dataset, seq_collate_dict

    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=str, default="../data",
                        help='data directory')
    parser.add_argument('--subset', type=str, default="Train",
                        help='whether to load Train/Valid/Test data')
    args = parser.parse_args()

    print("Loading data...")
    dataset = load_dataset(['acoustic', 'emotient', 'ratings'],
                           args.dir, args.subset, truncate=True,
                           item_as_dict=True)
    print("Building model...")
    model = MultiARLSTM(['acoustic', 'emotient'], [988, 31],
                      device=torch.device('cpu'))
    model.eval()
    print("Passing a sample through the model...")
    data, mask, lengths = seq_collate_dict([dataset[0]])
    target = data['ratings']
    out = model(data, mask, lengths, target=target).view(-1)
    print("Predicted valences:")
    for o in out:
        print("{:+0.3f}".format(o.item()))
