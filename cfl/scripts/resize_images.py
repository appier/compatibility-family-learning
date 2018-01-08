import argparse
import logging
import multiprocessing as mp
import os

from scipy.misc import imread, imsave, imresize
from tqdm import tqdm

from .utils import load_input_ids

logger = logging.getLogger(__name__)


def process_resize(args):
    src, dst, output_shape = args

    if os.path.exists(src):
        image = imread(src)
        image = imresize(image, output_shape)
        imsave(dst, image, 'jpeg')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image-dir', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--input-file', required=True)
    parser.add_argument(
        '--output-shape', default=(256, 256, 3), type=int, nargs='+')
    parser.add_argument('--n-threads', type=int, default=12)
    return parser.parse_args()


def resize(image_dir, output_dir, input_file, output_shape, n_threads):

    logger.info('get asins...')
    asins, _ = load_input_ids(input_file)
    os.makedirs(output_dir, exist_ok=True)

    resize_jobs = []
    for asin in asins:
        src = os.path.join(image_dir, asin + '.jpg')
        dst = os.path.join(output_dir, asin + '.jpg')
        resize_jobs.append((src, dst, output_shape))
    with mp.Pool(n_threads) as pool:
        for _ in tqdm(
                pool.imap_unordered(process_resize, resize_jobs),
                total=len(resize_jobs)):
            pass


def main():
    logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)
    args = parse_args()
    resize(**vars(args))


if __name__ == '__main__':
    main()
