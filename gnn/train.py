"""
CUDA_VISIBLE_DEVICES=1 python train.py --test-data cub
"""
import os
import torch
import time
import json
import random
import torch.nn.functional as F
from tqdm import tqdm

from dgfsl.utils.data import BatchMetaDataLoader
from utils import get_accuracy, get_dataset_helper, mean_confidence_interval, get_model_inputs, correct
from model import GnnNet


def train_val(args, model, meta_optimizer):
    assert args.record

    data_helper = get_dataset_helper(args.data_name)
    meta_train_dataset = data_helper(args.folder, shots=args.num_shots, ways=args.num_ways,
            shuffle=True, test_shots=args.test_shots, meta_train=True, download=args.download)
    meta_train_dataloader = BatchMetaDataLoader(meta_train_dataset, batch_size=args.batch_size,
            shuffle=True, num_workers=args.num_workers)

    meta_val_dataset = data_helper(args.folder, shots=args.num_shots, ways=args.num_ways,
        shuffle=True, test_shots=args.test_shots, meta_val=True, download=args.download)
    meta_val_dataloader = BatchMetaDataLoader(meta_val_dataset, batch_size=args.batch_size,
        shuffle=True, num_workers=args.num_workers)
    meta_val_iter = iter(meta_val_dataloader)

    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []
    best_val_acc = 0.0
    best_batch_num = 0
    best_state_dict = None

    with tqdm(meta_train_dataloader, total=args.num_batches) as pbar:
        for batch_idx, train_batch in enumerate(pbar):
            if batch_idx >= args.num_batches:
                break

            model.train()
            model.zero_grad()

            support_inputs, support_targets, query_inputs, query_targets = get_model_inputs(args, train_batch, args.data_name)
            train_loss = torch.tensor(0., device=args.device)
            accs = []

            for task_idx, (support_input, support_target, query_input,
                           query_target) in enumerate(zip(support_inputs, support_targets,
                                                             query_inputs, query_targets)):

                support_input = support_input.view(args.num_ways, -1, 3, 84, 84)
                query_input = query_input.view(args.num_ways, -1, 3, 84, 84)
                gnn_inputs = torch.cat([support_input, query_input], dim=1)

                scores, loss = model(gnn_inputs)
                correct_this, count_this, _ = correct(scores, loss, args.num_ways, 15)
                accs.append(correct_this/count_this)
                train_loss += loss

            accuracy, acc_std = get_accuracy(accs)
            train_loss.backward()
            meta_optimizer.step()
            pbar.set_postfix(accuracy='{0:.4f}'.format(accuracy))
            train_losses.append(train_loss.item())
            train_accs.append(accuracy)

            model.eval()

            try:
                val_batch = next(meta_val_iter)
            except StopIteration:
                del meta_val_iter
                del meta_val_dataloader
                meta_val_dataloader = BatchMetaDataLoader(meta_val_dataset, batch_size=args.batch_size,
                                                          shuffle=True, num_workers=args.num_workers)
                meta_val_iter = iter(meta_val_dataloader)
                val_batch = next(meta_val_iter)

            support_inputs, support_targets, query_inputs, query_targets = get_model_inputs(args, val_batch,
                                                                                            args.data_name)
            val_loss = torch.tensor(0., device=args.device)
            val_accuracy = []
            for task_idx, (support_input, support_target, query_input,
                           query_target) in enumerate(zip(support_inputs, support_targets,
                                                          query_inputs, query_targets)):

                support_input = support_input.view(args.num_ways, -1, 3, 84, 84)
                query_input = query_input.view(args.num_ways, -1, 3, 84, 84)
                gnn_inputs = torch.cat([support_input, query_input], dim=1)
                scores, loss = model(gnn_inputs)
                correct_this, count_this, _ = correct(scores, loss, args.num_ways, 15)
                val_accuracy.append(correct_this/count_this)
                val_loss += loss

            val_accuracy, acc_std = get_accuracy(val_accuracy)
            pbar.set_postfix(accuracy='{0:.4f}'.format(val_accuracy))
            val_losses.append(val_loss.item())
            val_accs.append(val_accuracy)

            if val_accuracy >= best_val_acc:
                save_model(args, model, 'best', batch_idx)
                best_batch_num = batch_idx
                best_val_acc = val_accuracy
                best_state_dict = model.state_dict()

    save_model(args, model, 'last', args.num_batches - 1)
    return best_state_dict, best_val_acc, best_batch_num, train_losses, train_accs, val_losses, val_accs


def save_model(args, model, tag, batch_idx):

    filename = os.path.join(args.record_folder, 'gnn_{0}_'
                                                '{1}shot_{2}way_{3}.pt'.format(args.data_name, args.num_shots,
                                                                               args.num_ways, tag))
    torch.save({
        'state_dict': model.state_dict(),
        'batch_idx': batch_idx
    }, filename)


def test(args, model):
    test_helper = get_dataset_helper(args.data_name)
    meta_test_dataset = test_helper(args.folder, shots=args.num_shots, ways=args.num_ways,
                                    shuffle=True, test_shots=args.test_shots, meta_test=True, download=args.download)
    meta_test_dataloader = BatchMetaDataLoader(meta_test_dataset, batch_size=args.batch_size,
                                               shuffle=True, num_workers=args.num_workers)
    accuracies = []

    with tqdm(meta_test_dataloader, total=args.num_exps) as pbar:
        for batch_idx, batch in enumerate(pbar):
            if batch_idx >= args.num_exps:
                break

            model.eval()
            support_inputs, support_targets, query_inputs, query_targets = get_model_inputs(args, batch, args.data_name)

            test_loss = torch.tensor(0., device=args.device)
            test_accuracy = []

            for task_idx, (support_input, support_target, query_input,
                           query_target) in enumerate(zip(support_inputs, support_targets,
                                                          query_inputs, query_targets)):

                support_input = support_input.view(args.num_ways, -1, 3, 84, 84)
                query_input = query_input.view(args.num_ways, -1, 3, 84, 84)
                gnn_inputs = torch.cat([support_input, query_input], dim=1)
                scores, loss = model(gnn_inputs)
                correct_this, count_this, _ = correct(scores, loss, args.num_ways, 15)
                test_accuracy.append(correct_this/count_this)
                test_loss += loss

            test_accuracy, h = get_accuracy(test_accuracy)
            pbar.set_postfix(accuracy='{0:.4f}'.format(test_accuracy))
            accuracies.append(test_accuracy)

    return accuracies


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser('Gnn for few-shot learning')
    # experimental settings
    parser.add_argument('--folder', type=str, default='./dataset',
        help='Path to the folder the data is downloaded to.')
    parser.add_argument('--data-name', type=str, default='cub', choices=['cub', 'miniimagenet', 'omniglot','cifar_fs'], help='specify multi for training with multiple domains.')
    parser.add_argument('--num-shots', type=int, default=5, choices=[1, 5, 10],
        help='Number of examples per class (k in "k-shot", default: 5).')
    parser.add_argument('--num-ways', type=int, default=5, choices=[5, 20],
        help='Number of classes per task (N in "N-way", default: 5).')
    parser.add_argument('--test-shots', type=int, default=15,
        help='Number of query examples per class (k in "k-shot", default: 15).')
    parser.add_argument('--batch-size', type=int, default=4,
        help='Number of tasks in a mini-batch of tasks (default: 4).')
    parser.add_argument('--num-batches', type=int, default=100,
        help='Number of batches the network is trained over (default: 10000).')
    parser.add_argument('--num-exps', type=int, default=50,
        help='Number of batches the network is tested over (default: 500). The final results will be the average of these batches.')
    # arguments of CNN
    parser.add_argument('--embedding-size', type=int, default=64,
        help='Dimension of the embedding/latent space (default: 64).')
    parser.add_argument('--hidden-size', type=int, default=64,
        help='Number of channels for each convolutional layer (default: 64).')
    # arguments of program
    parser.add_argument('--model-name', type=str, default='protonet',
        help='Name of the model.')    # ? need to change with model
    parser.add_argument('--num-workers', type=int, default=6,
        help='Number of workers for data loading (default: 6).')
    parser.add_argument('--download', action='store_true',
        help='Download the dataset in the data folder.')
    parser.add_argument('--record', type=bool, default=True,
        help='Record the results.')
    parser.add_argument('--use-cuda', type=bool, default=True,
        help='Use CUDA if available.')

    args = parser.parse_args()

    if args.record:
        cur_time = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime())
        args.record_folder = './Gnn_record_{}'.format(cur_time)
        os.makedirs(args.record_folder, exist_ok=True)
        with open(os.path.join(args.record_folder, 'args.json'), 'w') as f:
            json.dump(vars(args), f)

    args.device = torch.device('cuda' if args.use_cuda
                                         and torch.cuda.is_available() else 'cpu')

    model = GnnNet(args.num_ways, args.num_shots, args.test_shots)
    model.to(device=args.device)
    meta_optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    print('Meta Training ...')
    best_state_dict, best_val_acc, best_batch_num, train_losses, train_accs, val_losses, val_accs = train_val(args,
                                                                                                               model,
                                                                                                    meta_optimizer)
    print('Meta Testing ...')
    best_model = GnnNet(args.num_ways, args.num_shots, args.test_shots)
    best_model.load_state_dict(best_state_dict)
    best_model.to(device=args.device)
    test_accuracies = test(args, best_model)
    mean, h = mean_confidence_interval(test_accuracies)
    print('Testing Result: {0:.2f}+/-{1:.2f}'.format(mean * 100, h * 100))