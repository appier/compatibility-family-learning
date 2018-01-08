"""
split_meta.py

Split meta file to train / val / test
Split features.b, put unused items into train.
"""
import argparse
import logging
import os
import shutil

from ..input_data import dump_array
from ..input_data import load_features
from ..input_data import load_features_indices
from ..input_data import load_meta_lines

logger = logging.getLogger(__name__)


def load_items(path):
    with open(path) as infile:
        return {line.strip() for line in infile}


def load_pair_items(*paths):
    items = set()
    for path in paths:
        with open(path) as infile:
            for line in infile:
                id1, _, id2 = line.strip().split(' ')
                items.add(id1)
                items.add(id2)
    return items


def split_meta(feature_path, meta_path, data_dir, input_size=4096):
    # load items
    item_path = os.path.join(data_dir, 'items.txt')
    items = load_items(item_path)

    # load meta lines
    meta_lines = {}
    for item_id, lines in load_meta_lines(meta_path):
        if item_id not in meta_lines and item_id in items:
            meta_lines[item_id] = lines

    all_split_items = set()
    for split in ('train', 'val', 'test'):
        logger.info('generate %s meta...', split)
        paths = [
            os.path.join(data_dir, split, 'pairs_pos.txt'),
            os.path.join(data_dir, split, 'pairs_neg.txt'),
        ]
        split_items = load_pair_items(*paths)
        all_split_items.update(split_items)

        split_meta_path = os.path.join(data_dir, split, 'meta.txt')
        with open(split_meta_path, 'w') as outfile:
            for item_id in sorted(split_items):
                for line in meta_lines[item_id]:
                    outfile.write(line)

    # append to train
    train_feature_path = os.path.join(data_dir, 'train', 'features.b.raw')
    train_meta_path = os.path.join(data_dir, 'train', 'meta.txt')
    full_train_feature_path = os.path.join(data_dir, 'train', 'features.b')

    logger.info('copy train features...')
    shutil.copyfile(train_feature_path, full_train_feature_path)

    logger.info('append train features & meta...')
    train_feature_items = load_features_indices(
        train_feature_path, input_size=input_size)
    with open(train_meta_path, 'a') as meta_file, open(full_train_feature_path,
                                                       'ab') as feature_file:
        for item_id, feature in load_features(feature_path, input_size):
            if item_id in items and item_id not in all_split_items:
                all_split_items.add(item_id)
                if item_id not in train_feature_items:
                    feature_file.write(item_id.encode('ascii'))
                    dump_array(feature_file, feature)
                for line in meta_lines[item_id]:
                    meta_file.write(line)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('feature_path')
    parser.add_argument('meta_path')
    parser.add_argument('data_dir')
    args = parser.parse_args()

    split_meta(**vars(args))


if __name__ == '__main__':
    main()
