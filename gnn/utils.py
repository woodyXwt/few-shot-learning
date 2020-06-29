import numpy as np
import torch
import numpy as np
import scipy.stats
from dgfsl.datasets.helpers import miniimagenet, cub, omniglot, cifar_fs
from collections import OrderedDict

def one_hot(y, num_class):
    return torch.zeros((len(y), num_class)).scatter_(1, y.unsqueeze(1), 1)

def correct(scores, loss, n_way, n_query):
    y_query = np.repeat(range(n_way), n_query)
    topk_scores, topk_labels = scores.data.topk(1, 1, True, True)
    topk_ind = topk_labels.cpu().numpy()
    top1_correct = np.sum(topk_ind[:, 0] == y_query)
    return float(top1_correct), len(y_query), loss.item() * len(y_query)

def get_accuracy(acc_all):
    acc_all = np.asarray(acc_all)
    acc_mean = np.mean(acc_all)
    acc_std = np.std(acc_all)
    return acc_mean, acc_std

def get_dataset_helper(data_name):

    dataset_func = {
        'miniimagenet': miniimagenet,
        'omniglot': omniglot,
        'cub': cub,
        'cifar_fs': cifar_fs
    }
    return dataset_func[data_name]

def mean_confidence_interval(accuracies, confidence=0.95):
    n = len(accuracies)
    mean, standard_error = np.mean(accuracies), scipy.stats.sem(accuracies)
    h = standard_error * scipy.stats.t._ppf((1 + confidence) / 2, n - 1)
    # ~= 1.96 * standard_error
    return mean, h

def get_result_str(accuracies):
    mean, h = mean_confidence_interval(accuracies)
    result_str = '{0:.2f}; {1:.2f}'.format(mean * 100, h * 100)
    return result_str

def get_model_inputs(args, batch, data_name):

    if data_name == 'cub':
        support_inputs, support_targets = batch['train']
        query_inputs, query_targets = batch['test']
    elif data_name == 'miniimagenet':
        support_inputs, support_targets = batch['train']
        query_inputs, query_targets = batch['test']
    elif data_name == 'cifar_fs':
        support_inputs, support_targets = batch['train']
        query_inputs, query_targets = batch['test']
    elif data_name == 'omniglot':
        support_inputs, support_targets = batch['train']
        query_inputs, query_targets = batch['test']

    support_inputs = support_inputs.to(device=args.device)
    support_targets = support_targets.to(device=args.device)
    query_inputs = query_inputs.to(device=args.device)
    query_targets = query_targets.to(device=args.device)

    return support_inputs, support_targets, query_inputs, query_targets
