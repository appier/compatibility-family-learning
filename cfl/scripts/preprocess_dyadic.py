import argparse
import logging
import gzip
import json
import os

from collections import defaultdict

from tqdm import tqdm

logger = logging.getLogger(__name__)


def parse_meta(path, present_suffix, positive_ids):
    graph = set()
    images = defaultdict(set)
    full_urls = {}
    full_urls_of_asins = {}
    with gzip.open(path, 'r') as infile:
        for line in tqdm(infile):
            item = json.loads(line.decode('utf8'))
            if 'asin' in item and 'imUrl' in item:
                asin = item['asin']
                full_url = item['imUrl']
                url = full_url.split('/')[-1].rsplit('.', 1)[0]

                if url not in present_suffix and asin not in positive_ids:
                    continue

                full_urls_of_asins[asin] = full_url
                if url not in full_urls:
                    full_urls[url] = full_url
                else:
                    assert full_urls[url] == full_url

                if 'related' in item:
                    for rel in ('also_bought', 'bought_together'):
                        for target in item['related'].get(rel, []):
                            graph.add((asin, target))
                images[url].add(asin)

    return graph, images, full_urls, full_urls_of_asins


def parse(meta_path, data_path):
    splits = [
        ('train.txt', 'train.parsed.txt', 'train_ids.txt',
         'train_ids.parsed.txt'),
        ('val.txt', 'val.parsed.txt', 'val_ids.txt', 'val_ids.parsed.txt'),
        ('test.txt', 'test.parsed.txt', 'test_ids.txt', 'test_ids.parsed.txt'),
    ]

    logger.warning('load present ids...')
    present_suffix = set()
    for src, _, _, _ in splits:
        src_path = os.path.join(data_path, src)
        data = load_data(src_path)
        for x, y, _ in data:
            present_suffix.add(x)
            present_suffix.add(y)
    logger.warning('totally %d present suffix', len(present_suffix))

    logger.warning('load positive ids...')
    # prefer explicit present
    positive_ids = set()
    id_paths = ['train_ids.txt', 'val_ids.txt', 'test_ids.txt']
    for path in id_paths:
        path = os.path.join(data_path, path)
        with open(path) as infile:
            positive_ids.update(json.load(infile))
    logger.warning('totally %d positive ids', len(positive_ids))

    logger.warning('load graph...')
    (graph, image_suffix, full_urls, full_urls_of_asins) = parse_meta(
        meta_path, present_suffix, positive_ids)

    logger.warning('dump id pairs...')
    with open(os.path.join(data_path, 'all_id_pairs.txt'), 'w') as outfile:
        visited = set()
        for p in sorted(present_suffix):
            done = False
            ids = sorted(image_suffix[p])

            for asin in ids:
                if asin in positive_ids:
                    done = True
                    if asin not in visited:
                        outfile.write('{}\t{}\n'.format(asin, full_urls[p]))
                        visited.add(asin)

            if done:
                # prefer present
                image_suffix[p] = {
                    asin
                    for asin in ids if asin in positive_ids
                }
            else:
                # just choose one
                image_suffix[p] = {ids[0]}
                if ids[0] not in visited:
                    outfile.write('{}\t{}\n'.format(ids[0], full_urls[p]))
                    visited.add(ids[0])

        for asin in sorted(positive_ids - visited):
            visited.add(asin)
            outfile.write('{}\t{}\n'.format(asin, full_urls_of_asins[asin]))
    logger.warning('totally dump %d ids...', len(visited))

    logger.warning('dump ids...')
    for src, dst, src_ids, dst_ids in splits:
        src_path = os.path.join(data_path, src)
        dst_path = os.path.join(data_path, dst)

        src_ids_path = os.path.join(data_path, src_ids)
        dst_ids_path = os.path.join(data_path, dst_ids)

        data = load_data(src_path)
        ids = set()
        with open(dst_path, 'w') as outfile:
            for x, y, label in data:
                if label == '0':
                    asin1 = sorted(image_suffix[x])[0]
                    asin2 = sorted(image_suffix[y])[0]
                    outfile.write('{}\t{}\t{}\n'.format(asin1, asin2, label))
                    ids.add(asin1)
                    ids.add(asin2)
                else:
                    done = False
                    for asin1 in sorted(image_suffix[x]):
                        for asin2 in sorted(image_suffix[y]):
                            if (asin1, asin2) in graph:
                                outfile.write(
                                    '{}\t{}\t{}\n'.format(asin1, asin2, label))
                                done = True
                                ids.add(asin1)
                                ids.add(asin2)
                                break
                            elif (asin2, asin1) in graph:
                                outfile.write(
                                    '{}\t{}\t{}\n'.format(asin2, asin1, label))
                                done = True
                                ids.add(asin1)
                                ids.add(asin2)
                                break
                        if done:
                            break
                    assert done, '{} {} not present'.format(x, y)
        with open(src_ids_path) as infile, open(dst_ids_path, 'w') as outfile:
            ids.update(json.load(infile))
            json.dump(sorted(ids), outfile)


def load_data(path):
    data = []
    with open(path) as infile:
        for line in infile:
            x, y, label = line.strip().split()
            x = x.split('/')[-1].rsplit('.', 1)[0]
            y = y.split('/')[-1].rsplit('.', 1)[0]
            data.append((x, y, label))
    return data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--meta-path', required=True)
    parser.add_argument('--data-path', required=True)
    args = parser.parse_args()

    parse(**vars(args))


if __name__ == '__main__':
    main()
