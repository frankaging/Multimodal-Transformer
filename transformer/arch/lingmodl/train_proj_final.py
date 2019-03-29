"""Training code for synchronous multimodal LSTM model."""

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import sys, os, shutil
import argparse
import copy

import pandas as pd
import numpy as np
from scipy.stats import pearsonr
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from datasets import seq_collate_dict, load_dataset
from models import MultiLSTM, MultiEDLSTM, MultiARLSTM, MultiCNNLSTM
from multiTransformer import NLPTransformer

from random import shuffle
from operator import itemgetter
import pprint

import logging
logFilename = "./train_cnn.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(message)s",
    handlers=[
        logging.FileHandler(logFilename, 'w'),
        logging.StreamHandler()
    ])
logger = logging.getLogger()

def eval_ccc(y_true, y_pred):
    """Computes concordance correlation coefficient."""
    true_mean = np.mean(y_true)
    true_var = np.var(y_true)
    pred_mean = np.mean(y_pred)
    pred_var = np.var(y_pred)
    covar = np.cov(y_true, y_pred, bias=True)[0][1]
    ccc = 2*covar / (true_var + pred_var +  (pred_mean-true_mean) ** 2)
    return ccc

def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]

def generateTrainBatch(input_data, input_target, input_length, args, batch_size=25):
    # (data, target, mask, lengths)
    input_size = len(input_data)
    index = [i for i in range(0, input_size)]
    shuffle(index)
    shuffle_chunks = [i for i in chunks(index, batch_size)]
    for chunk in shuffle_chunks:
        data_chunk = [input_data[index] for index in chunk] # <- ~batch_size, x, y, z
        target_chunk = [input_target[index] for index in chunk] # <- ~batch_size, x
        length_chunk = [input_length[index] for index in chunk] # <- ~batch_size
        # print(length_chunk)

        max_length = max(length_chunk)

        combined_data = list(zip(data_chunk, length_chunk))
        combined_data.sort(key=itemgetter(1),reverse=True)
        combined_rating = list(zip(target_chunk, length_chunk))
        combined_rating.sort(key=itemgetter(1),reverse=True)
        data_sort = []
        target_sort = []
        length_sort = []
        for pair in combined_data:
            data_sort.append(pair[0])
            length_sort.append(pair[1])

        for pair in combined_rating:
            target_sort.append(pair[0])

        data_sort = torch.tensor(data_sort, dtype=torch.float)
        target_sort = torch.tensor(target_sort, dtype=torch.float)
        old_length_sort = copy.deepcopy(length_sort)
        length_sort = torch.tensor(length_sort)
        data_sort = data_sort[:,:max_length,:,:]
        target_sort = target_sort[:,:max_length]

        lstm_masks = torch.zeros(data_sort.size()[0], data_sort.size()[1], 1, dtype=torch.float)
        for i in range(lstm_masks.size()[0]):
            lstm_masks[i,:old_length_sort[i]] = 1
        # print(lstm_masks.size())
        # print(data_sort.size())
        # print(target_sort.size())
        # length_sort = torch.tensor(length_sort, dtype=torch.float)
        yield (data_sort, torch.unsqueeze(target_sort, dim=2), lstm_masks, old_length_sort)

def train(input_data, input_target, lengths, model, criterion, optimizer, epoch, args):
    model.train()
    data_num = 0
    loss = 0.0
    batch_num = 0
    # batch our data
    for (data, target, mask, lengths) in generateTrainBatch(input_data,
                                                            input_target,
                                                            lengths,
                                                            args):
        # send to device
        mask = mask.to(args.device)
        data = data.to(args.device)
        target = target.to(args.device)
        # lengths = lengths.to(args.device)
        # Run forward pass.
        output = model(data, lengths, mask)
        # Compute loss and gradients
        batch_loss = criterion(output, target)
        # Accumulate total loss for epoch
        loss += batch_loss
        # Average over number of non-padding datapoints before stepping
        batch_loss /= sum(lengths)
        batch_loss.backward()
        # Step, then zero gradients
        optimizer.step()
        optimizer.zero_grad()
        # Keep track of total number of time-points
        data_num += sum(lengths)
        logger.info('Batch: {:5d}\tLoss: {:2.5f}'.\
              format(batch_num, loss/data_num))
        batch_num += 1
    # Average losses and print
    loss /= data_num
    logger.info('---')
    logger.info('Epoch: {}\tLoss: {:2.5f}'.format(epoch, loss))
    return loss


def generateTrainBatchTest(input_data, input_length, args, batch_size=25):
    # (data, target, mask, lengths)
    input_size = len(input_data)
    index = [i for i in range(0, input_size)]
    shuffle(index)
    shuffle_chunks = [i for i in chunks(index, batch_size)]
    for chunk in shuffle_chunks:
        data_chunk = [input_data[index] for index in chunk] # <- ~batch_size, x, y, z
        length_chunk = [input_length[index] for index in chunk] # <- ~batch_size
        # print(length_chunk)

        max_length = max(length_chunk)

        combined_data = list(zip(data_chunk, length_chunk))
        combined_data.sort(key=itemgetter(1),reverse=True)
        data_sort = []
        length_sort = []
        for pair in combined_data:
            data_sort.append(pair[0])
            length_sort.append(pair[1])

        data_sort = torch.tensor(data_sort, dtype=torch.float)
        old_length_sort = copy.deepcopy(length_sort)
        length_sort = torch.tensor(length_sort)
        data_sort = data_sort[:,:max_length,:,:]

        lstm_masks = torch.zeros(data_sort.size()[0], data_sort.size()[1], 1, dtype=torch.float)
        for i in range(lstm_masks.size()[0]):
            lstm_masks[i,:old_length_sort[i]] = 1
        # print(lstm_masks.size())
        # print(data_sort.size())
        # print(target_sort.size())
        # length_sort = torch.tensor(length_sort, dtype=torch.float)
        yield (data_sort, lstm_masks, old_length_sort)

def evaluateTest(input_data, lengths, model, criterion, args, fig_path=None):
    model.eval()

    for (data, mask, lengths) in generateTrainBatchTest(input_data,
                                                        lengths,
                                                        args,
                                                        batch_size=1):
        # send to device
        mask = mask.to(args.device)
        data = data.to(args.device)
        # Run forward pass
        output = model(data, lengths, mask)
        return output

def evaluate(input_data, input_target, lengths, model, criterion, args, fig_path=None):
    model.eval()
    predictions = []
    data_num = 0
    loss, corr, ccc = 0.0, [], []
    count = 0

    local_best_output = []
    local_best_target = []
    local_best_index = 0
    index = 0
    local_best_ccc = -1
    for (data, target, mask, lengths) in generateTrainBatch(input_data,
                                                            input_target,
                                                            lengths,
                                                            args,
                                                            batch_size=1):
        # send to device
        mask = mask.to(args.device)
        data = data.to(args.device)
        target = target.to(args.device)
        # Run forward pass
        output = model(data, lengths, mask)
        # Compute loss
        loss += criterion(output, target)
        # Keep track of total number of time-points
        data_num += sum(lengths)
        # Compute correlation and CCC of predictions against ratings
        output = torch.squeeze(torch.squeeze(output, dim=2), dim=0).cpu().numpy()
        target = torch.squeeze(torch.squeeze(target, dim=2), dim=0).cpu().numpy()
        if count == 0:
            # print(output)
            # print(target)
            count += 1
        curr_ccc = eval_ccc(output, target)
        corr.append(pearsonr(output, target)[0])
        ccc.append(curr_ccc)
        index += 1
        if curr_ccc > local_best_ccc:
            local_best_output = output
            local_best_target = target
            local_best_index = index
            local_best_ccc = curr_ccc
    # Average losses and print
    loss /= data_num
    # Average statistics and print
    stats = {'corr': np.mean(corr), 'corr_std': np.std(corr),
             'ccc': np.mean(ccc), 'ccc_std': np.std(ccc), 'max_ccc': local_best_ccc}
    logger.info('Evaluation\tLoss: {:2.5f}\tCorr: {:0.3f}\tCCC: {:0.9f}'.\
          format(loss, stats['corr'], stats['ccc']))
    return predictions, loss, stats, (local_best_output, local_best_target, local_best_index)

def plot_predictions(dataset, predictions, metric, args, fig_path=None):
    """Plots predictions against ratings for representative fits."""
    # Select top 4 and bottom 4
    sel_idx = np.concatenate((np.argsort(metric)[-4:][::-1],
                              np.argsort(metric)[:4]))
    sel_metric = [metric[i] for i in sel_idx]
    sel_true = [dataset.orig['ratings'][i] for i in sel_idx]
    sel_pred = [predictions[i] for i in sel_idx]
    for i, (true, pred, m) in enumerate(zip(sel_true, sel_pred, sel_metric)):
        j, i = (i // 4), (i % 4)
        args.axes[i,j].cla()
        args.axes[i,j].plot(true, 'b-')
        args.axes[i,j].plot(pred, 'c-')
        args.axes[i,j].set_xlim(0, len(true))
        args.axes[i,j].set_ylim(-1, 1)
        args.axes[i,j].set_title("Fit = {:0.3f}".format(m))
    plt.tight_layout()
    plt.draw()
    if fig_path is not None:
        plt.savefig(fig_path)
    plt.pause(1.0 if args.test else 0.001)

def save_predictions(dataset, predictions, path):
    for p, seq_id in zip(predictions, dataset.seq_ids):
        df = pd.DataFrame(p, columns=['rating'])
        fname = "target_{}_{}_normal.csv".format(*seq_id)
        df.to_csv(os.path.join(path, fname), index=False)

def save_params(args, model, train_stats, test_stats):
    fname = 'param_hist.tsv'
    df = pd.DataFrame([vars(args)], columns=vars(args).keys())
    df = df[['modalities', 'batch_size', 'split', 'epochs', 'lr',
             'sup_ratio', 'base_rate']]
    for k in ['ccc_std', 'ccc']:
        v = train_stats.get(k, float('nan'))
        df.insert(0, 'train_' + k, v)
    for k in ['ccc_std', 'ccc']:
        v = test_stats.get(k, float('nan'))
        df.insert(0, 'test_' + k, v)
    df.insert(0, 'model', [model.__class__.__name__])
    df['embed_dim'] = model.embed_dim
    df['h_dim'] = model.h_dim
    df['attn_len'] = model.attn_len
    if type(model) is MultiARLSTM:
        df['ar_order'] = [model.ar_order]
    else:
        df['ar_order'] = [float('nan')]
    df.set_index('model')
    df.to_csv(fname, mode='a', header=(not os.path.exists(fname)), sep='\t')
        
def save_checkpoint(modalities, model, path):
    checkpoint = {'modalities': modalities, 'model': model.state_dict()}
    torch.save(checkpoint, path)

def load_checkpoint(path, device):
    checkpoint = torch.load(path, map_location='cpu')
    return checkpoint

def load_data(modalities, data_dir):
    print("Loading data...")
    # train_data = load_dataset(modalities, data_dir, 'Train',
    #                           base_rate=args.base_rate,
    #                           truncate=True, item_as_dict=True)
    test_data = load_dataset(modalities, data_dir, 'Test',
                             base_rate=args.base_rate,
                             truncate=True, item_as_dict=True)
    print("Done.")
    return test_data

def constructCNNInputTest(input_data, window_size=5, overlap_size=2):
    # input_data -> [num_vid, ds]
    # output_data -> (num_vid, num_window_in_vid, num_words_in_window, 300)
    CNNInput = []
    TargetOutput = []
    cc = 0

    for data in input_data:
        word_counter = 0
        current_time = 0.0
        videoInput = []
        windowInput = []
        videoEndOffset = data['linguistic_timer'][len(data['linguistic_timer'])-1][1]

        # print(data['linguistic'])

        out = []
        for word_vec in data['linguistic']:
            out2 = []
            for e in word_vec:
                if np.isnan(e):
                    out2.append(0)
                else:
                    out2.append(e)
            out.append(out2)

        # with overlap
        # for word_vec in out:
        #     offset = data['linguistic_timer'][word_counter][1] 
        #     if offset > current_time and current_time + window_size < videoEndOffset:
        #         index = word_counter
        #         while offset <= current_time + window_size:
        #             windowInput.append(out[index])
        #             index += 1
        #             if index < len(data['linguistic_timer']):
        #                 offset = data['linguistic_timer'][index][1]
        #         videoInput.append(windowInput)
        #         current_time += overlap_size
        #         windowInput = []
        #     word_counter += 1

        # # rating
        # WindowOutput = []
        # window_size_c = window_size/0.5
        # overlap_size_c = overlap_size/0.5
        # rating_sum = 0.0
        # for i in range(0, len(data['ratings']), int(overlap_size_c)):
        #     if i + window_size_c < len(data['ratings']):
        #         for j in range(i, int(i+window_size_c)):
        #             rating_sum += data['ratings'][j][0]
        #         WindowOutput.append((rating_sum*1.0/window_size_c))
        #         rating_sum = 0.0

        biggerWords = []
        smallerWords = []

        for word_vec in out:
            # print(sum(np.isnan(word_vec)))
            onset = data['linguistic_timer'][word_counter][0]
            offset = data['linguistic_timer'][word_counter][1]
            if offset <= current_time + window_size:
                windowInput.append(word_vec)
                smallerWords.append(data['linguistic_word'][word_counter])
            else:
                videoInput.append(windowInput)
                biggerWords.append(smallerWords)
                windowInput = [word_vec]
                smallerWords = [data['linguistic_word'][word_counter]]
                current_time += window_size
            word_counter += 1
        

        # WindowOutput = []
        # window_size_c = window_size/0.5
        # rating_sum = 0.0
        # for i in range(0, len(data['ratings'])):
        #     rating_sum += data['ratings'][i][0]
        #     if i != 0 and i%window_size_c == 0:
        #         WindowOutput.append((rating_sum*1.0/window_size_c))
        #         rating_sum = 0.0

        # rating_sum = []
        # for i in range(0, len(data['ratings'])):
        #     rating_sum.append(data['ratings'][i][0])
        #     if i != 0 and i%window_size_c == 0:
        #         WindowOutput.append(statistics.median(rating_sum))
        #         rating_sum = []

        # truncate to min length

        # print(cc, len(videoInput), len(WindowOutput))



        # minL = min(len(videoInput), len(WindowOutput))
        # videoInput = videoInput[:minL]
        # WindowOutput = WindowOutput[:minL]
        CNNInput.append(videoInput)
        # TargetOutput.append(WindowOutput)
        cc += 1
        
    return CNNInput, biggerWords

def constructCNNInput(input_data, window_size=5, overlap_size=2):
    # input_data -> [num_vid, ds]
    # output_data -> (num_vid, num_window_in_vid, num_words_in_window, 300)
    CNNInput = []
    TargetOutput = []
    cc = 0

    for data in input_data:
        word_counter = 0
        current_time = 0.0
        videoInput = []
        windowInput = []
        videoEndOffset = data['linguistic_timer'][len(data['linguistic_timer'])-1][1]

        # print(data['linguistic'])

        out = []
        for word_vec in data['linguistic']:
            out2 = []
            for e in word_vec:
                if np.isnan(e):
                    out2.append(0)
                else:
                    out2.append(e)
            out.append(out2)

        # with overlap
        # for word_vec in out:
        #     offset = data['linguistic_timer'][word_counter][1] 
        #     if offset > current_time and current_time + window_size < videoEndOffset:
        #         index = word_counter
        #         while offset <= current_time + window_size:
        #             windowInput.append(out[index])
        #             index += 1
        #             if index < len(data['linguistic_timer']):
        #                 offset = data['linguistic_timer'][index][1]
        #         videoInput.append(windowInput)
        #         current_time += overlap_size
        #         windowInput = []
        #     word_counter += 1

        # # rating
        # WindowOutput = []
        # window_size_c = window_size/0.5
        # overlap_size_c = overlap_size/0.5
        # rating_sum = 0.0
        # for i in range(0, len(data['ratings']), int(overlap_size_c)):
        #     if i + window_size_c < len(data['ratings']):
        #         for j in range(i, int(i+window_size_c)):
        #             rating_sum += data['ratings'][j][0]
        #         WindowOutput.append((rating_sum*1.0/window_size_c))
        #         rating_sum = 0.0


        for word_vec in out:
            # print(sum(np.isnan(word_vec)))
            onset = data['linguistic_timer'][word_counter][0]
            offset = data['linguistic_timer'][word_counter][1]
            if offset <= current_time + window_size:
                windowInput.append(word_vec)
            else:
                videoInput.append(windowInput)
                windowInput = [word_vec]
                current_time += window_size
            word_counter += 1
        

        WindowOutput = []
        window_size_c = window_size/0.5
        rating_sum = 0.0
        for i in range(0, len(data['ratings'])):
            rating_sum += data['ratings'][i][0]
            if i != 0 and i%window_size_c == 0:
                WindowOutput.append((rating_sum*1.0/window_size_c))
                rating_sum = 0.0

        # rating_sum = []
        # for i in range(0, len(data['ratings'])):
        #     rating_sum.append(data['ratings'][i][0])
        #     if i != 0 and i%window_size_c == 0:
        #         WindowOutput.append(statistics.median(rating_sum))
        #         rating_sum = []

        # truncate to min length

        # print(cc, len(videoInput), len(WindowOutput))



        minL = min(len(videoInput), len(WindowOutput))
        videoInput = videoInput[:minL]
        WindowOutput = WindowOutput[:minL]
        CNNInput.append(videoInput)
        TargetOutput.append(WindowOutput)
        cc += 1
        
    return CNNInput

def size4d(input):
    dim1 = len(input)
    dim2 = len(input[0])
    dim3 = len(input[0][0])
    dim4 = len(input[0][0][0])
    return dim1, dim2, dim3, dim4

def paddingCNNInputTest(input_data):
    # input_data -> (num_vid, num_window_in_vid, num_words_in_window, 300)
    # input_rating -> (num_vid, num_window_in_vid)
    # output_data -> (num_vid, num_window_in_vid, max_num_words_in_window, 300)
    output = []
    max_num_words = 0
    max_num_windows = 0
    num_windows_len = []
    for data in input_data:
        if max_num_windows < len(data):
            max_num_windows = len(data)
        num_windows_len.append(len(data))
        if max_num_words < max([len(w) for w in data]):
            max_num_words = max([len(w) for w in data])

    padVec = [0.0]*300
    for vid in input_data:
        vidNewTmp = []
        for wind in vid:
            windNew = [padVec] * max_num_words
            windNew[:len(wind)] = wind
            vidNewTmp.append(windNew)
        vidNew = [[padVec] * max_num_words]*max_num_windows
        vidNew[:len(vidNewTmp)] = vidNewTmp
        output.append(vidNew)

    return output, num_windows_len

def paddingCNNInput(input_data, input_rating):
    # input_data -> (num_vid, num_window_in_vid, num_words_in_window, 300)
    # input_rating -> (num_vid, num_window_in_vid)
    # output_data -> (num_vid, num_window_in_vid, max_num_words_in_window, 300)
    output = []
    max_num_words = 0
    max_num_windows = 0
    num_windows_len = []
    for data in input_data:
        if max_num_windows < len(data):
            max_num_windows = len(data)
        num_windows_len.append(len(data))
        if max_num_words < max([len(w) for w in data]):
            max_num_words = max([len(w) for w in data])

    padVec = [0.0]*300
    for vid in input_data:
        vidNewTmp = []
        for wind in vid:
            windNew = [padVec] * max_num_words
            windNew[:len(wind)] = wind
            vidNewTmp.append(windNew)
        vidNew = [[padVec] * max_num_words]*max_num_windows
        vidNew[:len(vidNewTmp)] = vidNewTmp
        output.append(vidNew)

    rating_output = []
    # pad ratings
    for rating in input_rating:
        ratingNew = [0]*max_num_windows
        ratingNew[:len(rating)] = rating
        rating_output.append(ratingNew)

    return output, rating_output, num_windows_len

def main(args):
    # Fix random seed
    torch.manual_seed(1)
    torch.cuda.manual_seed(1)
    np.random.seed(1)
    # Convert device string to torch.device
    args.device = (torch.device(args.device) if torch.cuda.is_available()
                   else torch.device('cpu'))
    evaluating = True
    if evaluating:
        # load the model
        model_path = os.path.join("./lstm_save", "transformer_best.pth")
        checkpoint = load_checkpoint(model_path, args.device)
        args.modalities = ['linguistic']
        dims = {'linguistic': 128}
        args.window_embed_size = 128    
        model = MultiCNNLSTM(device=args.device)
        model.load_state_dict(checkpoint['model'])
        criterion = nn.MSELoss(reduction='sum')
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)

        # load the data
        test_data = load_data(args.modalities, args.data_dir)
        # print(test_data[0]['linguistic'])
        CNNInput, biggerWords = constructCNNInputTest(test_data)
        cnnInput, num_windows_len = paddingCNNInputTest(CNNInput)

        with torch.no_grad():
            pred = evaluateTest(cnnInput, num_windows_len, model, criterion, args)
            pred_li = pred.view(-1).tolist()
            new_words_window = []
            for window in biggerWords:
                ww = []
                for w in window:
                    ww.append(w[0].strip())
                new_words_window.append(",".join(ww))
            print(biggerWords)
            print(new_words_window)
            print(pred_li)
            # print(dict(zip(biggerWords, pred_li)))
            
        return
        

    args.modalities = ['linguistic']

    # Load data for specified modalities
    train_data, test_data = load_data(args.modalities, args.data_dir)

    # print(train_data[0]['linguistic'])
    CNNInput, TargetOutput = constructCNNInput(train_data)
    # print(CNNInput)
    cnnInput, targetOutput, num_windows_len = paddingCNNInput(CNNInput, TargetOutput)

    # print(CNNInput)

    # Load test data
    CNNInput_test, TargetOutput_test = constructCNNInput(test_data)
    cnnInput_test, targetOutput_test, num_windows_len_test = \
        paddingCNNInput(CNNInput_test, TargetOutput_test)

    # cnnInput = (117, 39, 33, 300)
    # targetOutput = (117, 39)

    # construct model
    args.modalities = ['linguistic']
    dims = {'linguistic': 128}
    args.window_embed_size = 128    
    model = MultiCNNLSTM(device=args.device)

    criterion = nn.MSELoss(reduction='sum')
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    # lambda1 = lambda epoch: epoch // 30
    # lambda2 = lambda epoch: 1e-6
    # scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=[lambda2])
    # Train and save best model
    best_ccc = -1
    single_best_ccc = -1
    for epoch in range(1, args.epochs+1):
        print('---')
        # scheduler.step()
        train(cnnInput, targetOutput, num_windows_len,
              model, criterion, optimizer, epoch, args)
        if epoch % args.eval_freq == 0:
            with torch.no_grad():
                pred, loss, stats, (local_best_output, local_best_target, local_best_index) =\
                    evaluate(cnnInput_test, targetOutput_test, num_windows_len_test,
                             model, criterion, args)
            if stats['ccc'] > best_ccc:
                best_ccc = stats['ccc']
                path = os.path.join(args.save_dir, "transformer_best.pth") 
                save_checkpoint(args.modalities, model, path)
            if stats['max_ccc'] > single_best_ccc:
                single_best_ccc = stats['max_ccc']
                logger.info('===single_max_predict===')
                logger.info(local_best_output)
                logger.info(local_best_target)
                logger.info(local_best_index)
                logger.info('===end single_max_predict===')
            logger.info('CCC_STATS\tSINGLE_BEST: {:0.9f}\tBEST: {:0.9f}'.\
            format(single_best_ccc, best_ccc))
        # # Save checkpoints
        # if epoch % args.save_freq == 0:
        #     path = os.path.join(args.save_dir,
        #                         "epoch_{}.pth".format(epoch)) 
        #     save_checkpoint(args.modalities, model, path)

    # Save final model
    # path = os.path.join(args.save_dir, "last.pth") 
    # save_checkpoint(args.modalities, model, path)

    # Save command line flags, model params and performance statistics
    # save_params(args, model, dict(), best_stats)
    
    return best_ccc

if __name__ == "__main__":
    # global global_best_ccc
    # global global_single_vid_best_ccc
    parser = argparse.ArgumentParser()
    parser.add_argument('--modalities', type=str, default=None, nargs='+',
                        help='input modalities (default: all')
    parser.add_argument('--batch_size', type=int, default=10, metavar='N',
                        help='input batch size for training (default: 10)')
    parser.add_argument('--split', type=int, default=1, metavar='N',
                        help='sections to split each video into (default: 1)')
    parser.add_argument('--epochs', type=int, default=1000, metavar='N',
                        help='number of epochs to train (default: 1000)')
    parser.add_argument('--lr', type=float, default=1e-4, metavar='LR',
                        help='learning rate (default: 1e-6)')
    parser.add_argument('--sup_ratio', type=float, default=0.5, metavar='F',
                        help='teacher-forcing ratio (default: 0.5)')
    parser.add_argument('--base_rate', type=float, default=2.0, metavar='N',
                        help='sampling rate to resample to (default: 2.0)')
    parser.add_argument('--log_freq', type=int, default=5, metavar='N',
                        help='print loss N times every epoch (default: 5)')
    parser.add_argument('--eval_freq', type=int, default=1, metavar='N',
                        help='evaluate every N epochs (default: 1)')
    parser.add_argument('--save_freq', type=int, default=10, metavar='N',
                        help='save every N epochs (default: 10)')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='device to use (default: cuda:0 if available)')
    parser.add_argument('--visualize', action='store_true', default=False,
                        help='flag to visualize predictions (default: false)')
    parser.add_argument('--normalize', action='store_true', default=False,
                        help='whether to normalize inputs (default: false)')
    parser.add_argument('--test', action='store_true', default=False,
                        help='evaluate without training (default: false)')
    parser.add_argument('--load', type=str, default=None,
                        help='path to trained model (either resume or test)')
    parser.add_argument('--data_dir', type=str, default="../data",
                        help='path to data base directory')
    parser.add_argument('--save_dir', type=str, default="./lstm_save",
                        help='path to save models and predictions')
    args = parser.parse_args()
    main(args)