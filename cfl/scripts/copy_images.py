import argparse
import logging
import os
import multiprocessing as mp

from collections import defaultdict

from tqdm import tqdm

from ..s3_utils import connect_s3
from ..s3_utils import load_dir_jls
from ..s3_utils import copy_file
from ..s3_utils import file_exists
from .utils import load_input_ids

logger = logging.getLogger(__name__)


def process_copy(args):
    src, dst = args
    conn = connect_s3()
    assert file_exists(conn, src)
    copy_file(conn, src, dst)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--items-store', required=True)
    parser.add_argument('--images-store', required=True)
    parser.add_argument('--output-path', required=True)
    parser.add_argument('--input-file', required=True)
    parser.add_argument('--n-threads', type=int, default=12)
    return parser.parse_args()


def split(images_store, items_store, output_path, input_file, n_threads):

    logger.info('get asins...')
    asins, image_paths = load_input_ids(input_file)
    visited = set()

    exts = set()
    logger.info('load paths...')
    conn = connect_s3()
    items = list(load_dir_jls(conn, os.path.join(items_store, 'amazon_item')))

    copy_jobs = []
    for item in tqdm(items):
        if ('images' in item and len(item['images']) > 0 and 'asin' in item and
                item['asin'] not in visited):
            for image_item in item.get('images', []):
                url = image_item['url']
                path = image_item['path']
                src = os.path.join(images_store, path)
                ext = '.' + src.split('.')[-1]
                exts.add(ext)

                for asin in image_paths[url]:
                    if asin not in visited and asin in asins:
                        visited.add(asin)
                        dst = os.path.join(output_path, asin + ext)
                        copy_jobs.append((src, dst))

    logger.info('%d ids not found', len(asins) - len(visited))
    for asin in (asins - visited):
        logger.info('= %s', asin)

    logger.info('%d exts = %s', len(exts), ','.join(exts))

    with mp.Pool(n_threads) as pool:
        for _ in tqdm(
                pool.imap_unordered(process_copy, copy_jobs),
                total=len(copy_jobs)):
            pass


def main():
    logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)
    args = parse_args()
    split(**vars(args))


if __name__ == '__main__':
    main()
