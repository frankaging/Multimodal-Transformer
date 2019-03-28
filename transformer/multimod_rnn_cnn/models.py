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
    def __init__(self, mods, 
                 word_embed_sizeL=300, word_embed_sizeA=988, word_embed_sizeE=20, cnn_rnn_embedE=128, 
                 cnn_embed_sizeL=128, cnn_embed_sizeA=256, cnn_embed_sizeE=64,
                 window_embed_size=128,
                 kL=5, kA=10, kE=3,
                 device=torch.device('cuda:0')):
        super(MultiCNNLSTM, self).__init__()
        self.mods = mods
        self.CNN_L = CNN(word_embed_sizeL, cnn_embed_sizeL, kL)
        self.CNN_A = CNN(word_embed_sizeA, cnn_embed_sizeA, kA)
        self.CNN_RNN = nn.LSTM(word_embed_sizeE, cnn_rnn_embedE, 1, batch_first=True)
        self.CNN_E = CNN(cnn_rnn_embedE, cnn_embed_sizeE, kE)
        self.Highway_L = Highway(cnn_embed_sizeL)
        self.Highway_A = Highway(cnn_embed_sizeA)
        self.Highway_E = Highway(cnn_embed_sizeE)
        LSTM_EMB = 0
        if 'linguistic' in self.mods:
            LSTM_EMB += cnn_embed_sizeL
        if 'acoustic' in self.mods:
            LSTM_EMB += cnn_embed_sizeA
        if 'emotient' in self.mods:
            LSTM_EMB += cnn_embed_sizeE

        self.LSTM = NLPTransformer(LSTM_EMB)
        self.dropout = nn.Dropout(p=0.3)
        # Store module in specified device (CUDA/CPU)
        self.device = (device if torch.cuda.is_available() else
                       torch.device('cpu'))
        self.to(self.device)

    def forward(self, inputs, length, window_length_sort, mask=None):
        '''
        inputs = (batch_size, 39, 33, 300)
        '''
        # print(inputs[0])
        combined_outputs = []
        if 'linguistic' in self.mods:
            # print("!")
            combined_outputs_L = []
            for x in torch.split(inputs[0], 1, 0):
                x = torch.squeeze(x, 0) # input -> (39, 33, 300)
                # print(x.size())
                cnnOut = self.CNN_L(x.permute(0, 2, 1)) # -> (39, 128)
                # print(cnnOut)
                x_highway = self.Highway_L(cnnOut)
                x_word_emb = self.dropout(x_highway)
                combined_outputs_L.append(x_word_emb)
            combined_outputs_L = torch.stack(combined_outputs_L, dim=0) # -> (batch_size, seq_l, 128)
            combined_outputs.append(combined_outputs_L)
        if 'acoustic' in self.mods:
            # print("!!")
            combined_outputs_A = []
            for x in torch.split(inputs[1], 1, 0):
                x = torch.squeeze(x, 0) # input -> (39, 33, 300)
                # print(x.size())
                cnnOut = self.CNN_A(x.permute(0, 2, 1)) # -> (39, 128)
                # print(cnnOut)
                x_highway = self.Highway_A(cnnOut)
                x_word_emb = self.dropout(x_highway)
                combined_outputs_A.append(x_word_emb)
                # print(x_word_emb)
            combined_outputs_A = torch.stack(combined_outputs_A, dim=0) # -> (batch_size, seq_l, 128)
            combined_outputs.append(combined_outputs_A)
        if 'emotient' in self.mods:
            # print("!!!")
            combined_outputs_E = []
            i = 0
            
            # print(len(window_length_sort))
            # print(len(window_length_sort[0]))
            for x in torch.split(inputs[2], 1, 0):
                # TODO: just taking the avg now, no CNN
                # input <- (39, ~max window, 20)
                # output <- (39, 20)
                x = torch.squeeze(x, 0) # input -> (39, ~max window, 20)
                # cnnOut = self.CNN_E(x.permute(0, 2, 1)) # -> (39, 128)
                # x_highway = self.Highway_E(cnnOut)
                # x_word_emb = self.dropout(x_highway)
                x_rnn_list = []
                # split over the RNN batch -> 39
                j = 0
                for x_rnn in torch.split(x, 1, 0): # <- (1, ~max window, 20)
                    max_window_length = x_rnn.size()[1]
                    # print(x_rnn.size())
                    x_rnn_length = window_length_sort[i][j]
                    # print(x_rnn_length)
                    x_rnn_pack = pack_padded_sequence(x_rnn, [x_rnn_length], batch_first=True)
                    h0 = torch.zeros(1, 1, 128).to(self.device)
                    c0 = torch.zeros(1, 1, 128).to(self.device)
                    h, _ = self.CNN_RNN(x_rnn_pack, (h0, c0))
                    h, _ = pad_packed_sequence(h, batch_first=True) # <- (1, ~max window, 128)
                    # pad h so that have 150 as length
                    if h.size()[1] < max_window_length:
                        h = torch.nn.functional.pad(h, (0, 0, 0, max_window_length-h.size()[1]))
                    x_rnn_list.append(torch.squeeze(h, 0))
                    j += 1
                # print(len(x_rnn_list))
                # print(x_rnn_list[0].size())
                x_rnn_stack = torch.stack(x_rnn_list, dim=0) # -> (39, 150, 128)
                print(x_rnn_stack.size())
                cnnOut = self.CNN_E(x_rnn_stack.permute(0, 2, 1)) # -> (39, 128)
                x_highway = self.Highway_E(cnnOut)
                x_word_emb = self.dropout(x_highway)
                combined_outputs_E.append(x_word_emb)
                i += 1
            # print(combined_outputs_E)
            combined_outputs_E = torch.stack(combined_outputs_E, dim=0) # -> (batch_size, seq_l, 128)
            combined_outputs.append(combined_outputs_E)
            # print(combined_outputs_E)
        if len(combined_outputs) > 1:
            combined_outputs = torch.cat(combined_outputs, 2)
        else:
            combined_outputs = combined_outputs[0]
        # print(combined_outputs)
        predict = self.LSTM(combined_outputs, mask, length)
        # print(predict.tolist())
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
