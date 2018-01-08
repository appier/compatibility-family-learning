"""
Convert Polyvore data to monomer-like data
"""
import argparse
import struct
import logging
import os
import random

from io import BytesIO

import numpy as np

from tqdm import tqdm
from scipy.misc import imread, imsave, imresize

from ..input_data import load_meta_lines
from ..input_data import load_features

logger = logging.getLogger(__name__)

source_cates = {
    ('Clothing Shoes & Jewelry', 'Women', 'Clothing'),
    ('Clothing Shoes & Jewelry', 'Men', 'Clothing'),
}
target_cates = {('Clothing Shoes & Jewelry', 'Women',
                 'Shoes'), ('Clothing Shoes & Jewelry', 'Men', 'Shoes'),
                ('Clothing Shoes & Jewelry', 'Men',
                 'Watches'), ('Clothing Shoes & Jewelry', 'Women', 'Watches')}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-dir', required=True)
    parser.add_argument('--image-dir', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--preserve', action='store_true')
    parser.add_argument('--seed', type=int, default=633)
    return parser.parse_args()


def main():
    logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)
    args = parse_args()
    convert(**vars(args))


def convert(input_dir, image_dir, output_dir, seed, preserve):
    random.seed(seed)
    for split in ('train', 'val', 'test'):
        split_path = os.path.join(output_dir, split)
        os.makedirs(split_path, exist_ok=True)

        source_ids_set = set()
        target_ids_set = set()

        meta_path = os.path.join(input_dir, split, 'meta.txt')
        meta_lines = {}
        for item_id, lines in load_meta_lines(meta_path):
            is_target = is_source = False
            for meta_line in lines[1:]:
                cates = tuple([cate.strip() for cate in meta_line.split(',')])
                for n in range(len(cates)):
                    if cates[:n + 1] in source_cates:
                        is_source = True
                    if cates[:n + 1] in target_cates:
                        is_target = True
            if (is_source or is_target) and not (is_source and is_target):
                if is_source:
                    source_ids_set.add(item_id)
                if is_target:
                    target_ids_set.add(item_id)
                meta_lines[item_id] = lines

        logger.warning('%s: source: %d target: %d', split,
                       len(source_ids_set), len(target_ids_set))

        source_path = os.path.join(split_path, 'source.txt')
        target_path = os.path.join(split_path, 'target.txt')
        source_ids_lst = sorted(source_ids_set)
        target_ids_lst = sorted(target_ids_set)
        random.shuffle(source_ids_lst)
        random.shuffle(target_ids_lst)

        tasks = [(source_path, source_ids_lst), (target_path, target_ids_lst)]
        for path, split_ids in tasks:
            with open(path, 'w') as outfile:
                for asin in split_ids:
                    outfile.write(asin + '\n')
        split_meta_path = os.path.join(split_path, 'meta.txt')
        with open(split_meta_path, 'w') as outfile:
            for asin in source_ids_lst:
                for line in meta_lines[asin]:
                    outfile.write(line)
            for asin in target_ids_lst:
                for line in meta_lines[asin]:
                    outfile.write(line)

        ids = source_ids_set | target_ids_set
        feature_path = os.path.join(split_path, 'features.b')
        latents_path = os.path.join(input_dir, split, 'features.b')
        with open(feature_path, 'wb') as outfile:
            for asin, latent in tqdm(load_features(latents_path, 1024)):
                if asin in ids:
                    image_path = os.path.join(image_dir, asin + '.jpg')
                    with open(image_path, 'rb') as image_file:
                        outfile.write(asin.encode('ascii'))
                        image = imresize(imread(image_file), (64, 64, 3))
                        image_buf = BytesIO()
                        imsave(image_buf, image, 'png')
                        image = image_buf.getvalue()

                        latent_buf = BytesIO()
                        np.savez_compressed(latent_buf, data=latent)
                        latent = latent_buf.getvalue()

                        outfile.write(struct.pack('<i', len(image)))
                        outfile.write(struct.pack('<i', len(latent)))
                        outfile.write(image)
                        outfile.write(latent)

        # determine pairs
        pairs = [set(), set()]
        for i, name in enumerate(['pairs_pos.txt', 'pairs_neg.txt']):
            pair_path = os.path.join(input_dir, split, name)
            total = 0
            with open(pair_path) as infile:
                for line in infile:
                    tokens = line.strip().split()
                    if tokens[2] in source_ids_set and tokens[0] in target_ids_set:
                        tokens[0], tokens[2] = tokens[2], tokens[0]
                    if tokens[0] in source_ids_set and tokens[2] in target_ids_set:
                        pairs[i].add((tokens[0], tokens[2]))
                        total += 1
                    if (not preserve or split == 'val'
                        ) and i > 0 and len(pairs[i]) >= len(pairs[i - 1]):
                        break

        all_pairs = pairs[0] | pairs[1]

        for name in ('pairs_pos.txt', 'pairs_neg.txt', 'pairs_all.txt'):
            pair_path = os.path.join(input_dir, split, name)
            output_path = os.path.join(split_path, name)
            total = 0
            with open(pair_path) as infile, open(output_path, 'w') as outfile:
                for line in infile:
                    tokens = line.strip().split()
                    if tokens[2] in source_ids_set and tokens[0] in target_ids_set:
                        tokens[0], tokens[2] = tokens[2], tokens[0]
                    if (tokens[0], tokens[2]) in all_pairs:
                        total += 1
                        outfile.write('{}\n'.format(' '.join(tokens)))
            logger.warning('%s total = %d', pair_path, total)


if __name__ == '__main__':
    main()
