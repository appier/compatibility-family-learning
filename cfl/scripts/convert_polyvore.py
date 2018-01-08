"""
Convert Polyvore data to monomer-like data
"""
import argparse
import struct
import logging
import os
import random

from collections import defaultdict

from tqdm import tqdm
from sklearn.model_selection import train_test_split

from ..input_data import load_meta_lines

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-dir', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument(
        '--pos-percent',
        type=float,
        default=0.2,
        help='top fav_count percentage to be treated as positive')
    parser.add_argument(
        '--neg-sample-rate',
        type=int,
        default=4,
        help='number of random samples to be treated as negative')
    parser.add_argument(
        '--val-percent',
        type=float,
        default=0.2,
        help='percentage of pairs and items to be put in val')
    parser.add_argument(
        '--test-percent',
        type=float,
        default=0.2,
        help='percentage of pairs and items to be put in test')
    parser.add_argument('--seed', type=int, default=633)
    return parser.parse_args()


def main():
    logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)
    args = parse_args()
    convert(**vars(args))


def convert(input_dir, output_dir, pos_percent, neg_sample_rate, val_percent,
            test_percent, seed):
    # get available asins by listing image_dir
    logger.warning('get asins...')
    image_dir = os.path.join(input_dir, 'images')
    available_asins = {name[:-4] for name in os.listdir(image_dir)}

    cate_asins = []
    asin_cates = {}
    for i in range(3):
        cate_path = os.path.join(input_dir, 'cate_{}.txt'.format(i))
        with open(cate_path) as infile:
            cate_ids = {line.strip() for line in infile}
            cate_asins.append(cate_ids)
            for asin in cate_ids:
                asin_cates[asin] = i

    total_cate_num = sum(len(cate_ids) for cate_ids in cate_asins)

    logger.warning('loading meta...')
    meta_path = os.path.join(input_dir, 'meta.txt')
    meta_lines = {}
    for item_id, lines in tqdm(load_meta_lines(meta_path)):
        if item_id not in meta_lines and item_id in available_asins:
            meta_lines[item_id] = lines

    assert len(available_asins) == len(meta_lines), 'lacking some meta!!!'
    assert total_cate_num == len(meta_lines), 'lacking some cate!!!'

    logger.warning('loading outfits...')
    outfits = []
    with open(os.path.join(input_dir, 'outfits.txt')) as infile:
        for line in infile:
            fav_count, asins = line.strip().split('\t')
            asins = asins.split(' ')
            outfits.append((fav_count, asins))

    # bottom ranked first
    outfits.sort()

    total_num = len(outfits)
    pos_offset = int(total_num * (1. - pos_percent))

    pos_outfits = outfits[pos_offset:]

    directions = [
        ('top_to_other', ([0], [1, 2])),
        ('bottom_to_other', ([1], [0, 2])),
        ('shoe_to_other', ([2], [0, 1])),
    ]

    for name, (sources, targets) in directions:
        convert_direction(
            name=name,
            sources=sources,
            targets=targets,
            input_dir=input_dir,
            output_dir=output_dir,
            cate_asins=cate_asins,
            asin_cates=asin_cates,
            pos_outfits=pos_outfits,
            neg_sample_rate=neg_sample_rate,
            meta_lines=meta_lines,
            available_asins=available_asins,
            val_percent=val_percent,
            test_percent=test_percent,
            seed=seed)


def convert_direction(name, sources, targets, input_dir, output_dir,
                      cate_asins, asin_cates, pos_outfits, neg_sample_rate,
                      meta_lines, available_asins, val_percent, test_percent,
                      seed):
    random.seed(seed)
    root_dir = os.path.join(output_dir, name)
    image_dir = os.path.join(input_dir, 'images')
    latent_dir = os.path.join(input_dir, 'latents')

    source_ids_set = set()
    target_ids_set = set()
    for source in sources:
        source_ids_set |= cate_asins[source]
    for target in targets:
        target_ids_set |= cate_asins[target]
    source_ids = sorted(source_ids_set)
    target_ids = sorted(target_ids_set)

    train_source_ids, test_source_ids = train_test_split(
        source_ids, test_size=test_percent, random_state=seed)

    val_percent = val_percent / (1. - test_percent)

    train_source_ids, val_source_ids = train_test_split(
        train_source_ids, test_size=val_percent, random_state=seed)

    if set(sources) == set(targets):
        logger.warning('all to all, use same split')
        train_target_ids = train_source_ids
        val_target_ids = val_source_ids
        test_target_ids = test_source_ids
    else:
        train_target_ids, test_target_ids = train_test_split(
            target_ids, test_size=test_percent, random_state=seed)

        train_target_ids, val_target_ids = train_test_split(
            train_target_ids, test_size=val_percent, random_state=seed)

    id_splits = {
        'train': (set(train_source_ids), set(train_target_ids)),
        'val': (set(val_source_ids), set(val_target_ids)),
        'test': (set(test_source_ids), set(test_target_ids)),
    }

    splits = ('train', 'val', 'test')

    logger.warning('generate %s dataset / %d source / %d target...', name,
                   len(source_ids), len(target_ids))

    for split in splits:
        split_path = os.path.join(root_dir, split)
        os.makedirs(split_path, exist_ok=True)

        split_source_ids, split_target_ids = id_splits[split]
        split_source_ids_s, split_target_ids_s = sorted(
            split_source_ids), sorted(split_target_ids)
        split_meta_path = os.path.join(split_path, 'meta.txt')
        with open(split_meta_path, 'w') as outfile:
            for asin in split_source_ids_s:
                for line in meta_lines[asin]:
                    outfile.write(line)
            for asin in split_target_ids_s:
                for line in meta_lines[asin]:
                    outfile.write(line)

        not_found = set()
        skipped = 0
        split_pairs = {}
        pos_num = neg_num = 0
        for _, items in pos_outfits:
            items = list(items)
            random.shuffle(items)
            for source_item in items:
                for target_item in items:
                    if (source_item != target_item and
                            source_item in split_source_ids and
                            target_item in split_target_ids):
                        if ((source_item, target_item) not in split_pairs and
                            (target_item, source_item) not in split_pairs and
                                asin_cates[source_item] !=
                                asin_cates[target_item]):
                            if (source_item not in available_asins or
                                    target_item not in available_asins):
                                skipped += 1
                                not_found.update({source_item, target_item})
                            elif (source_item, target_item) not in split_pairs:
                                split_pairs[(source_item, target_item)] = 1
                                pos_num += 1
        for _ in range(len(split_pairs) * neg_sample_rate):
            neg_source = random.choice(split_source_ids_s)
            neg_target = random.choice(split_target_ids_s)
            if asin_cates[neg_source] == asin_cates[neg_target]:
                continue
            if ((neg_source, neg_target) not in split_pairs and
                (neg_target, neg_source) not in split_pairs):
                split_pairs[(neg_source, neg_target)] = 0
                neg_num += 1

        counts = defaultdict(int)
        for a, b in split_pairs:
            counts[(asin_cates[a], asin_cates[b])] += 1

        for (a, b), c in sorted(counts.items()):
            logger.warning('%s: %d -> %d = %d', split, a, b, c)

        not_found -= available_asins
        logger.warning(
            '%s: %d source / %d target / %d + %d = %d pairs: %d skipped / %d asins not found',
            split,
            len(split_source_ids),
            len(split_target_ids), pos_num, neg_num,
            len(split_pairs), skipped, len(not_found))

        pos_path = os.path.join(split_path, 'pairs_pos.txt')
        neg_path = os.path.join(split_path, 'pairs_neg.txt')
        all_path = os.path.join(split_path, 'pairs_all.txt')

        with open(pos_path,
                  'w') as posfile, open(neg_path, 'w') as negfile, open(
                      all_path, 'w') as outfile:
            for (asin1, asin2), label in sorted(split_pairs.items()):
                if label == 1:
                    label_file = posfile
                else:
                    label_file = negfile
                label_file.write('{} match {}\n'.format(asin1, asin2))
                outfile.write('{} match {} {}\n'.format(asin1, asin2, label))

        ids = sorted(split_source_ids | split_target_ids)
        random.shuffle(ids)
        feature_path = os.path.join(split_path, 'features.b')
        with open(feature_path, 'wb') as outfile:
            for asin in tqdm(ids):
                image_path = os.path.join(image_dir, asin + '.png')
                latent_path = os.path.join(latent_dir, asin + '.png')
                with open(image_path, 'rb') as image_file, open(
                        latent_path, 'rb') as latent_file:
                    outfile.write(asin.encode('ascii'))
                    image = image_file.read()
                    latent = latent_file.read()
                    outfile.write(struct.pack('<i', len(image)))
                    outfile.write(struct.pack('<i', len(latent)))
                    outfile.write(image)
                    outfile.write(latent)

        source_path = os.path.join(split_path, 'source.txt')
        target_path = os.path.join(split_path, 'target.txt')
        split_source_ids_lst = sorted(split_source_ids)
        split_target_ids_lst = sorted(split_target_ids)
        random.shuffle(split_source_ids_lst)
        random.shuffle(split_target_ids_lst)

        tasks = [(source_path, split_source_ids_lst), (target_path,
                                                       split_target_ids_lst)]
        for path, split_ids in tasks:
            with open(path, 'w') as outfile:
                for asin in split_ids:
                    outfile.write(asin + '\n')


if __name__ == '__main__':
    main()
