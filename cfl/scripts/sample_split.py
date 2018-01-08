import argparse
import logging
import os

from tqdm import tqdm
from scipy.misc import imread, imsave

logger = logging.getLogger(__name__)


def main(sample_dir, cut_width):
    split_dir = os.path.join(sample_dir, 'split')
    os.makedirs(split_dir, exist_ok=True)

    for name in tqdm(os.listdir(sample_dir)):
        if name.endswith('.png'):
            image = imread(os.path.join(sample_dir, name))
            image1 = image[:, :cut_width]
            image2 = image[:, cut_width:]
            imsave(os.path.join(split_dir, name[:-3] + '.h.png'), image1)
            imsave(os.path.join(split_dir, name[:-3] + '.p.png'), image2)


def start():
    log_format = '%(asctime)s [%(levelname)-5.5s] [%(name)s]  %(message)s'
    logging.basicConfig(format=log_format, level=logging.WARNING)
    args = parse_args()
    main(**vars(args))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sample-dir', required=True)
    parser.add_argument('--cut-width', type=int, default=28)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    start()
