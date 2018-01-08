import argparse
import logging
import os

import numpy as np

from tqdm import tqdm
from scipy.misc import imsave, imread


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-dir', required=True)
    parser.add_argument('--disco-dir', required=True)
    parser.add_argument('--output-dir', required=True)
    return parser.parse_args()


def main():
    logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)
    args = parse_args()
    convert(**vars(args))


def convert(input_dir, disco_dir, output_dir):
    for split in ('train', 'val', 'test'):
        pairs = []
        with open(os.path.join(input_dir, split, 'pairs_pos.txt')) as infile:
            for line in infile:
                asin1, _, asin2 = line.strip().split()
                pairs.append((asin1, asin2))
        split_path = os.path.join(output_dir, split)
        os.makedirs(split_path, exist_ok=True)

        src_img_path = os.path.join(disco_dir, split, 'A')
        tgt_img_path = os.path.join(disco_dir, split, 'B')
        for i, (asin1, asin2) in tqdm(enumerate(pairs)):
            img1 = imread(os.path.join(src_img_path, asin1 + '.png'))
            img2 = imread(os.path.join(tgt_img_path, asin2 + '.png'))
            image = np.concatenate([img1, img2], 1)
            imsave(
                os.path.join(split_path, '{}_{}.png'.format(asin1, asin2)),
                image)


if __name__ == '__main__':
    main()
