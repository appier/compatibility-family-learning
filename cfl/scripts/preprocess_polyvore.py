import argparse
import logging
import os

from io import BytesIO
from multiprocessing.pool import ThreadPool

from scipy.misc import imread, imsave, imresize
from tqdm import tqdm

from ..s3_utils import connect_s3
from ..s3_utils import load_dir_jls

from . import polyvore_categories

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--items-store', required=True)
    parser.add_argument('--image-dir', required=True)
    parser.add_argument('--image-shape', type=int, nargs='+', default=(64, 64))
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--skip-images', action='store_true')
    parser.add_argument('--n-threads', type=int, default=10)
    return parser.parse_args()


def main():
    logging.basicConfig(
        format=r'%(asctime)s [%(levelname)s] %(message)s', level=logging.INFO)
    args = parse_args()
    preprocess(**vars(args))


def preprocess(items_store, image_dir, output_dir, image_shape, skip_images,
               n_threads):
    allowed_categories = [
        polyvore_categories.top_categories,
        polyvore_categories.bottom_categories,
        polyvore_categories.shoe_categories,
    ]

    conn = connect_s3()
    items = {}
    item_categories = {}
    logging.info('load items...')
    for item in load_dir_jls(conn, os.path.join(items_store, 'polyvore_item')):
        product_id = int(item['product_id'])
        if not product_id or product_id in items:
            # must not appear before
            continue

        if len(item['images']) == 0 or not item['images'][0].get('path'):
            # must have images
            continue

        item_category = None
        category_seq = None
        for i, cate in enumerate(allowed_categories):
            item_category = polyvore_categories.first_category(
                item['categories'], cate)
            category_seq = i
            if item_category is not None:
                break

        if item_category is None:
            # must be of category
            continue

        image_hash = item['images'][0]['path'].split('/')[-1].split('.')[0]
        image_path = os.path.join(image_dir, image_hash + '.jpg')
        if not os.path.exists(image_path):
            # must have image
            continue

        items[product_id] = (product_id, item_category, image_path, item)
        item_categories[product_id] = category_seq

    logging.info('totally %d items', len(items))

    out_image_dir = os.path.join(output_dir, 'images')
    os.makedirs(out_image_dir, exist_ok=True)

    def resize_images(args):
        product_id, image_path = args
        if skip_images:
            out_path = os.path.join(out_image_dir,
                                    '{:010d}.png'.format(product_id))
            if os.path.exists(out_path):
                return product_id, True
            else:
                return product_id, None

        else:
            with open(image_path, 'rb') as infile:
                infile = BytesIO(infile.read())
                image = imread(infile)
            if (len(image.shape) == 3 and image.shape[2] == 3 and
                    image.shape[0] > image_shape[0] and
                    image.shape[1] > image_shape[1]):
                image = imresize(image, image_shape)
                return product_id, image
            else:
                return product_id, None

    p = ThreadPool(n_threads)

    tasks = [(item[0], item[2]) for item in items.values()]
    jobs = p.imap_unordered(resize_images, tasks)

    pass_ids = set()
    logging.info('save images...')
    for product_id, image in tqdm(jobs, total=len(items)):
        if image is not None:
            if not skip_images:
                out_path = os.path.join(out_image_dir,
                                        '{:010d}.png'.format(product_id))
                imsave(out_path, image)
            pass_ids.add(product_id)

    logging.info('totally %d items with correct images', len(pass_ids))
    items = {
        product_id: item
        for product_id, item in items.items() if product_id in pass_ids
    }

    logging.info('store meta...')
    with open(os.path.join(output_dir, 'meta.txt'), 'w') as outfile:
        for product_id, item in sorted(items.items()):
            outfile.write('{:010d}\n'.format(product_id))
            outfile.write(' {}\n'.format(','.join(item[1])))

    outfits = []
    seen = set()
    for outfit in load_dir_jls(conn,
                               os.path.join(items_store,
                                            'polyvore_outfit_set')):
        if outfit['url'] in seen:
            continue
        seen.add(outfit['url'])

        outfit_items = set([url.split('=')[-1] for url in outfit['items']])
        outfit_items = [int(asin) for asin in outfit_items]

        outfit_items = [
            product_id for product_id in outfit_items if product_id in items
        ]

        # only leave ok outfits
        all_cates = set()
        for product_id in outfit_items:
            all_cates.add(item_categories[product_id])

        if len(all_cates) > 1:
            outfit['items'] = outfit_items
            outfits.append(outfit)

    logging.info('totally %d outfits with > 1 selected items', len(outfits))

    # store three types
    for i in range(3):
        out_path = os.path.join(output_dir, 'cate_{}.txt'.format(i))
        with open(out_path, 'w') as outfile:
            for product_id in items:
                if item_categories[product_id] == i:
                    outfile.write('{:010d}\n'.format(product_id))

    # store pairs
    out_path = os.path.join(output_dir, 'outfits.txt')
    with open(out_path, 'w') as outfile:
        for outfit in outfits:
            fav_count = outfit['fav_count']
            if fav_count == 'Like':
                fav_count = 0
            else:
                fav_count = int(fav_count.replace(',', ''))

            outfit_items = [
                '{:010d}'.format(product_id) for product_id in outfit['items']
            ]
            outfile.write('{}\t{}\n'.format(fav_count, ' '.join(outfit_items)))


if __name__ == '__main__':
    main()
