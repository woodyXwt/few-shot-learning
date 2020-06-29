'''

CUDA_VISIBLE_DEVICES=2 python search.py
'''
import os
import torch
import json
from tqdm import tqdm
from pandas import DataFrame
import pandas as pd
import time

from dgfsl.utils.data import BatchMetaDataLoader

from utils import get_result_str
from train import train_val, test

from model import GnnNet

import torch


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser('Prototypical Networks')
    # experimental settings
    parser.add_argument('--folder', type=str, default='./dataset',
        help='Path to the folder the data is downloaded to.')
    parser.add_argument('--data-name', type=str, default='cub',choices=['cub', 'flowers',
        'sun','miniimagenet', 'tieredimagenet'], help='specify multi for training with multiple domains.')

    parser.add_argument('--num-shots', type=int, default=5, choices=[1, 5, 10],
        help='Number of examples per class (k in "k-shot", default: 5).')
    parser.add_argument('--num-ways', type=int, default=5, choices=[5, 20],
        help='Number of classes per task (N in "N-way", default: 5).')
    parser.add_argument('--test-shots', type=int, default=15,
        help='Number of query examples per class (k in "k-shot", default: 15).')
    parser.add_argument('--batch-size', type=int, default=4,
        help='Number of tasks in a mini-batch of tasks (default: 4).')
    parser.add_argument('--num-batches', type=int, default=10000,
        help='Number of batches the network is trained over (default: 10000).')
    parser.add_argument('--num-exps', type=int, default=500,
        help='Number of batches the network is tested over (default: 500). The final results will be the average of these batches.')
    # arguments of CNN
    parser.add_argument('--embedding-size', type=int, default=64,
        help='Dimension of the embedding/latent space (default: 64).')
    parser.add_argument('--hidden-size', type=int, default=64,
        help='Number of channels for each convolutional layer (default: 64).')
    parser.add_argument('--model-name', type=str, default='protonet',
        help='Name of the model.')
    parser.add_argument('--num-workers', type=int, default=6,
        help='Number of workers for data loading (default: 6).')
    parser.add_argument('--download', action='store_true',
        help='Download the dataset in the data folder.')
    parser.add_argument('--record', type=bool, default=True,
        help='Record the results.')
    parser.add_argument('--use-cuda', type=bool, default=True,
        help='Use CUDA if available.')

    args = parser.parse_args()
    record_root = None

    if args.record:
        cur_time = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime())
        record_root = './GNN_record_{}'.format(cur_time)
        os.makedirs(record_root, exist_ok=True)
        with open(os.path.join(record_root, 'args.json'), 'w') as f:
            json.dump(vars(args), f)

    args.device = torch.device('cuda' if args.use_cuda
        and torch.cuda.is_available() else 'cpu')

    index_values = [
        'test_acc',
        'best_batch_num',
        'best_train_acc',
        'best_train_loss',
        'best_val_acc',
        'best_val_loss'
    ]

    for data in ['cub', 'miniimagenet', 'omniglot', 'cifar_fs']:
        args.data_name = data
        if args.record:
            args.record_folder = os.path.join(record_root, '{}'.format(args.data_name))
            os.makedirs(args.record_folder, exist_ok=True)
        test_record = {}
        for num_shots in [1, 5]:
            print('Training for {} under {}-shots ...'.format(args.data_name, num_shots))
            args.num_shots = num_shots

            model = GnnNet(args.num_ways, args.num_shots, args.test_shots)
            model.to(device=args.device)
            meta_optimizer = torch.optim.Adam(model.parameters(), lr=1e-3) 

            hp_str = 'shots={}'.format(str(args.num_shots))

            best_state_dict, best_val_acc, best_batch_num, train_losses, train_accs, val_losses, val_accs = train_val(args, model, meta_optimizer)

            best_model = GnnNet(args.num_ways, args.num_shots, args.test_shots)
            best_model.load_state_dict(best_state_dict)
            best_model.to(device=args.device)
            test_accuracies = test(args, best_model)
                    
            test_record_data = [
                get_result_str(test_accuracies),
                str(best_batch_num),
                str(train_accs[best_batch_num]),
                str(train_losses[best_batch_num]),
                str(val_accs[best_batch_num]),
                str(val_losses[best_batch_num])
               ]

            test_record[hp_str] = test_record_data
        
        print('test_recoed:', test_record)

        if args.record:
            cur_time = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime())
            test_record_file = os.path.join(args.record_folder, 'test_record_{0}_{1}.csv'.format(args.data_name, cur_time))
            # test_record_file = os.path.join(args.record_folder, 'test_record_{0}shot_{1}.csv'.format(str(num_shots), cur_time))    
            DataFrame(test_record, index=index_values).to_csv(test_record_file)