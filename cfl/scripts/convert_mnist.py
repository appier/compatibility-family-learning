"""
Convert MNIST data to monomer data
"""
import argparse
import logging
import os
import random

import numpy as np

from scipy.misc import imresize

from ..input_data import SemiMNIST
from ..input_data import dump_array

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--data-path', default='data/MNIST_data')
    parser.add_argument(
        '--output-shape', type=int, nargs='+', default=(28, 28, 1))
    parser.add_argument('--labeled-percent', type=float, default=0.05)
    parser.add_argument('--all-digits', action='store_true')
    parser.add_argument('--symmetric', action='store_true', help='symmetric')
    parser.add_argument('--resample', action='store_true', help='resample')
    parser.add_argument(
        '--all-random',
        action='store_true',
        help='each base digit only sample'
        'either negative or positive but not both'
        'for each sample-rate')
    parser.add_argument('--sample-rate', type=int, default=1)
    parser.add_argument('--seed', type=int, default=633)
    args = parser.parse_args()
    assert len(args.output_shape) == 3

    convert(**vars(args))


def convert(output_dir, output_shape, labeled_percent, all_digits, data_path,
            symmetric, resample, all_random, sample_rate, seed):
    data_type = 'diffone' if not all_digits else 'diffone_all'
    data_splits = [
        ('train', SemiMNIST(
            data_type=data_type,
            path=data_path,
            split='train',
            labeled_percent=labeled_percent,
            all_random=all_random,
            sample_rate=sample_rate,
            symmetric=symmetric,
            resample=resample,
            seed=seed)),
        ('val', SemiMNIST(
            data_type=data_type,
            path=data_path,
            split='validation',
            all_random=all_random,
            sample_rate=sample_rate,
            labeled_percent=labeled_percent,
            symmetric=symmetric,
            resample=resample,
            seed=seed)),
        ('test', SemiMNIST(
            data_type=data_type,
            path=data_path,
            split='test',
            all_random=all_random,
            sample_rate=sample_rate,
            symmetric=symmetric,
            resample=resample,
            labeled_percent=1.0,
            seed=seed)),
    ]

    start_num = 0

    for split, data in data_splits:
        split_path = os.path.join(output_dir, split)
        os.makedirs(split_path, exist_ok=True)

        # dump features & categories
        images = data.mnist._images
        labels = data.mnist._labels
        logger.warning('%s: %d images', split, images.shape[0])
        if tuple(output_shape[:2]) != (28, 28):
            images = np.array([
                imresize(image.reshape((28, 28)), output_shape[:2])
                for image in images
            ]).reshape((-1,
                        output_shape[0] * output_shape[1])).astype(np.float32)
        images /= 255.

        fea_path = os.path.join(split_path, 'features.b')
        meta_path = os.path.join(split_path, 'meta.txt')

        with open(fea_path, 'wb') as outfile_fea, open(meta_path,
                                                       'w') as outfile_meta:
            for i in range(images.shape[0]):
                seq_num = start_num + i

                product_id = '{}{:09x}'.format(labels[i], seq_num)
                outfile_fea.write(product_id.encode('ascii'))
                dump_array(outfile_fea, images[i])

                digit_type = 'bottom' if labels[i] < 5 else 'top'
                outfile_meta.write(
                    '{}\n digits,{}\n'.format(product_id, digit_type))

        pos_path = os.path.join(split_path, 'pairs_pos.txt')
        neg_path = os.path.join(split_path, 'pairs_neg.txt')
        all_path = os.path.join(split_path, 'pairs_all.txt')

        tasks = [
            (pos_path, data.pairs_pos, 1),
            (neg_path, data.pairs_neg, 0),
        ]
        all_pairs = []
        for path, edges, label in tasks:
            with open(path, 'w') as outfile:
                for i in range(edges.shape[0]):
                    edge = edges[i]
                    label1 = labels[edge[0]]
                    label2 = labels[edge[1]]
                    id1 = edge[0] + start_num
                    id2 = edge[1] + start_num
                    outfile.write('{}{:09x} match {}{:09x}\n'.format(
                        label1, id1, label2, id2))
                    all_pairs.append((label1, id1, label2, id2, label))

        random.seed(seed)
        random.shuffle(all_pairs)
        with open(all_path, 'w') as outfile:
            for label_from, edge_from, label_to, edge_to, label in all_pairs:
                outfile.write('{}{:09x} match {}{:09x} {}\n'.format(
                    label_from, edge_from, label_to, edge_to, label))

        start_num += images.shape[0]


if __name__ == '__main__':
    main()
