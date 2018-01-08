"""
Convert Dyadic data to monomer-like data
"""
import argparse
import json
import logging
import os
import random

import numpy as np

from tqdm import tqdm
from sklearn.model_selection import train_test_split

from ..input_data import load_meta_lines
from ..input_data import dump_array

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-dir', required=True)
    parser.add_argument('--meta-path', required=True)
    parser.add_argument('--latent-dir', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--sample-val-rate', type=float, default=0.0)
    parser.add_argument('--seed', default=633)
    return parser.parse_args()


def main():
    logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)
    args = parse_args()
    convert(**vars(args))


def convert(input_dir, meta_path, latent_dir, output_dir, sample_val_rate,
            seed):
    # get available asins by listing latent_dir
    logger.warning('get asins...')
    available_asins = {name[:-4] for name in os.listdir(latent_dir)}

    logger.warning('loading meta...')
    meta_lines = {}
    for item_id, lines in tqdm(load_meta_lines(meta_path)):
        if item_id not in meta_lines and item_id in available_asins:
            meta_lines[item_id] = lines

    assert len(available_asins) == len(meta_lines), 'lacking some meta!!!'

    splits = ('train', 'val', 'test')

    pairs = {}
    logger.warning('loading pairs...')
    for split in splits:
        split_pairs = []
        not_found = set()
        skipped = 0

        pairs_path = os.path.join(input_dir, '{}.parsed.txt'.format(split))
        with open(pairs_path) as infile:
            for line in infile:
                asin1, asin2, label = line.strip().split('\t')
                if (asin1 not in available_asins or
                        asin2 not in available_asins):
                    skipped += 1
                    not_found.update({asin1, asin2})
                else:
                    split_pairs.append((asin1, asin2, label))

        not_found -= available_asins
        logger.warning('load %d pairs for %s: %d skipped / %d asin not found',
                       len(split_pairs), split, skipped, len(not_found))
        pairs[split] = split_pairs

    if sample_val_rate > 0.0:
        random.seed(seed)
        train = pairs['train']
        train = sorted(set(train))
        labels = [int(label) for _, _, label in train]
        train, val = train_test_split(
            train,
            test_size=sample_val_rate,
            stratify=labels,
            random_state=seed)

        pairs['val'].extend(val)
        val_set = set()
        for a, b, _ in val:
            val_set.add(a)
            val_set.add(b)
        len_train = len(train)
        train = [
            p for p in train if p[0] not in val_set and p[1] not in val_set
        ]

        len_pos = len([1 for p in train if p[2] == '1'])
        len_neg = len([1 for p in train if p[2] == '0'])
        len_train = len(train)
        len_pos_2 = len([1 for p in train if p[2] == '1'])
        len_neg_2 = len([1 for p in train if p[2] == '0'])
        logger.warning('%d ids in val, train %d -> %d (%d+%d) -> %d (%d+%d)',
                       len(val_set), len_train, len_train, len_pos, len_neg,
                       len(train), len_pos_2, len_neg_2)
        pairs['train'] = train

    for split in splits:
        logger.warning('%s: save %d pairs', split, len(pairs[split]))
        split_path = os.path.join(output_dir, split)
        os.makedirs(split_path, exist_ok=True)

        pos_path = os.path.join(split_path, 'pairs_pos.txt')
        neg_path = os.path.join(split_path, 'pairs_neg.txt')
        all_path = os.path.join(split_path, 'pairs_all.txt')

        with open(pos_path,
                  'w') as posfile, open(neg_path, 'w') as negfile, open(
                      all_path, 'w') as outfile:
            for asin1, asin2, label in pairs[split]:
                if label == '1':
                    label_file = posfile
                else:
                    label_file = negfile
                label_file.write('{} match {}\n'.format(asin1, asin2))
                outfile.write('{} match {} {}\n'.format(asin1, asin2, label))

        ids_path = os.path.join(input_dir, '{}_ids.parsed.txt'.format(split))
        logger.info('loading %s', ids_path)
        with open(ids_path) as infile:
            ids = set(json.load(infile))
        not_found = (ids - available_asins)
        if not_found:
            logger.warning('%d asins not found in %s: %s',
                           len(not_found), split, ','.join(not_found))

        ids &= available_asins
        for asin1, asin2, _ in pairs[split]:
            ids.add(asin1)
            ids.add(asin2)
        logger.warning('%s: save %d images', split, len(ids))
        ids = sorted(ids)
        random.seed(seed)
        random.shuffle(ids)
        feature_path = os.path.join(split_path, 'features.b')
        with open(feature_path, 'wb') as outfile:
            for asin in tqdm(ids):
                latent_path = os.path.join(latent_dir, asin + '.npy')
                with open(latent_path, 'rb') as latent_file:
                    outfile.write(asin.encode('ascii'))
                    latent = np.load(latent_file).flatten()
                    dump_array(outfile, latent)

        logger.warning('%s: save %d meta', split, len(ids))

        split_meta_path = os.path.join(split_path, 'meta.txt')
        with open(split_meta_path, 'w') as outfile:
            for asin in ids:
                for line in meta_lines[asin]:
                    outfile.write(line)


if __name__ == '__main__':
    main()
