import argparse
import logging
import os

from tqdm import tqdm
from scipy.misc import imsave

from ..input_data import yield_double_images


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-dir', required=True)
    parser.add_argument('--output-dir', required=True)
    return parser.parse_args()


def main():
    logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)
    args = parse_args()
    convert(**vars(args))


def convert(input_dir, output_dir):
    for split in ('train', 'val', 'test'):
        source_ids = set()
        with open(os.path.join(input_dir, split, 'source.txt')) as infile:
            for line in infile:
                asin = line.strip()
                source_ids.add(asin)

        target_ids = set()
        with open(os.path.join(input_dir, split, 'target.txt')) as infile:
            for line in infile:
                asin = line.strip()
                target_ids.add(asin)

        src_path = os.path.join(output_dir, split, 'A')
        tgt_path = os.path.join(output_dir, split, 'B')
        os.makedirs(src_path, exist_ok=True)
        os.makedirs(tgt_path, exist_ok=True)

        feature_path = os.path.join(input_dir, split, 'features.b')
        for asin, image in tqdm(yield_double_images(feature_path)):
            if asin in source_ids:
                imsave(os.path.join(src_path, asin + '.png'), image)
            elif asin in target_ids:
                imsave(os.path.join(tgt_path, asin + '.png'), image)


if __name__ == '__main__':
    main()
