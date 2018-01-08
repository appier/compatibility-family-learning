"""
Evaluate on test data selected by val data
"""
import argparse
import logging
import os

import numpy as np

from sklearn.metrics import roc_auc_score
from collections import defaultdict

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path', nargs='+', required=True)
    parser.add_argument('--predict-paths', nargs='+', required=True)
    parser.add_argument('--select-auc', action='store_true')
    parser.add_argument('--avg', action='store_true')
    parser.add_argument('--auc-model', action='store_true')
    parser.add_argument('--only-larger', type=int)
    parser.add_argument('--name', default='model')
    args = parser.parse_args()

    evaluate(**vars(args))


def load_pairs(path):
    p = []
    with open(path) as infile:
        for line in infile:
            ts = line.strip().split()
            x = ts[0]
            y = ts[2]
            v = float(ts[3]) if len(ts) >= 4 else 0.0
            p.append((x, y, v))
    return p


def load_data_pairs(path, only_larger=None):
    data_pairs = {}
    for split in ('train', 'val', 'test'):
        split_pairs = {}
        pos_pairs = load_pairs(os.path.join(path, split, 'pairs_pos.txt'))
        neg_pairs = load_pairs(os.path.join(path, split, 'pairs_neg.txt'))
        if only_larger:
            counts = defaultdict(int)
            for a, b, _ in pos_pairs:
                counts[a] += 1
            pos_pairs = [p for p in pos_pairs if counts[p[0]] > only_larger]
            neg_pairs = [p for p in neg_pairs if counts[p[0]] > only_larger]
        split_pairs['pos_pairs'] = pos_pairs
        split_pairs['neg_pairs'] = neg_pairs
        data_pairs[split] = split_pairs
        # logger.warning('%s: %d + %d', split, len(pos_pairs), len(neg_pairs))
    return data_pairs


def evaluate_data_set(data_pairs, predict_path, auc_model):

    if auc_model:
        predict_paths = {
            'train': os.path.join(predict_path, 'predict_train.txt'),
            'val': os.path.join(predict_path, 'predict_val.txt'),
            'test': os.path.join(predict_path, 'predict.txt'),
        }
    else:
        predict_paths = {
            'train': os.path.join(predict_path, 'predict_train_acc.txt'),
            'val': os.path.join(predict_path, 'predict_val_acc.txt'),
            'test': os.path.join(predict_path, 'predict_acc.txt'),
        }

    results = {}
    for split in ('train', 'val', 'test'):
        path = predict_paths[split]
        if split == 'train' and not os.path.exists(path):
            results[split] = {
                'accuracy': -1.,
                'error': -1.,
                'auc': -1.,
            }

        else:
            pred_pairs = load_pairs(path)
            pred_pairs = {(x, y): v for x, y, v in pred_pairs}

            true_pairs = data_pairs[split]

            result = evaluate_accuracy(true_pairs['pos_pairs'],
                                       true_pairs['neg_pairs'], pred_pairs)
            results[split] = result

    return results


def evaluate_accuracy(pos_pairs, neg_pairs, pred_pairs):

    y_true = []
    y_score = []
    for x, y, _ in pos_pairs:
        y_true.append(1)
        y_score.append(pred_pairs[(x, y)])
    for x, y, _ in neg_pairs:
        y_true.append(0)
        y_score.append(pred_pairs[(x, y)])
    accuracy, error = evaluate_accuracy_by_th(y_true, y_score)
    result = {
        'accuracy': accuracy,
        'error': error,
        'auc': roc_auc_score(y_true, y_score),
        'y_true': y_true,
        'y_score': y_score
    }
    return result


def evaluate_accuracy_by_th(y_true, y_score, th=0.0):
    correct = wrong = 0
    for label, score in zip(y_true, y_score):
        if label > 0:
            if score > th:
                correct += 1
            else:
                wrong += 1
        else:
            if score > th:
                wrong += 1
            else:
                correct += 1
    accuracy = correct / (wrong + correct)
    error = wrong / (wrong + correct)
    return accuracy, error


def select_best_result(results, select_auc):
    best_result = None
    for result in results:
        if best_result is None:
            best_result = result
        elif select_auc and result['val']['auc'] > best_result['val']['auc']:
            best_result = result
        elif not select_auc and result['val']['accuracy'] > best_result['val']['accuracy']:
            best_result = result
    return best_result


def average_result(results):
    avg_result = {}
    for split in ('train', 'val', 'test'):
        avg_result[split] = {'error': [], 'auc': []}
        for result in results:
            avg_result[split]['error'].append(result[split]['error'])
            avg_result[split]['auc'].append(result[split]['auc'])
        avg_result[split]['error_std'] = np.std(avg_result[split]['error'])
        avg_result[split]['auc_std'] = np.std(avg_result[split]['auc'])
        avg_result[split]['error'] = np.mean(avg_result[split]['error'])
        avg_result[split]['auc'] = np.mean(avg_result[split]['auc'])

    return avg_result


def print_result(result, name, avg):
    if avg:
        print(
            '{:.2%}+-{:.2%}\t{:.2%}+-{:.2%}\t{:.2%}+-{:.2%}\t{:.2%}+-{:.2%}\t{:.2%}+-{:.2%}\t{:.2%}+-{:.2%}\t{}'.
            format(result['train']['error'], result['train']['error_std'],
                   result['val']['error'], result['val']['error_std'], result[
                       'test']['error'], result['test']['error_std'],
                   result['train']['auc'], result['train']['auc_std'], result[
                       'val']['auc'], result['val']['auc_std'], result['test'][
                           'auc'], result['test']['auc_std'], name))
    else:
        print('{:.2%}\t{:.2%}\t{:.2%}\t{:.2%}\t{:.2%}\t{:.2%}\t{}'.format(
            result['train']['error'], result['val']['error'], result['test'][
                'error'], result['train']['auc'], result['val']['auc'], result[
                    'test']['auc'], name))


def evaluate(data_path, predict_paths, select_auc, name, avg, auc_model,
             only_larger):
    if len(data_path) == 1:
        data_pairs = load_data_pairs(data_path[0], only_larger)
        results = []
        for predict_path in predict_paths:
            results.append(
                evaluate_data_set(data_pairs, predict_path, auc_model))
    else:
        assert len(data_path) == len(predict_paths)
        results = []
        for a_data_path, predict_path in zip(data_path, predict_paths):
            data_pairs = load_data_pairs(a_data_path, only_larger)
            results.append(
                evaluate_data_set(data_pairs, predict_path, auc_model))

    if avg:
        result = average_result(results)
    else:
        result = select_best_result(results, select_auc)
    print_result(result, name, avg)


if __name__ == '__main__':
    main()
