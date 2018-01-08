from collections import defaultdict


def load_input_ids(path):
    ids = set()
    image_paths = defaultdict(set)
    with open(path) as infile:
        for line in infile:
            asin, url = line.strip().split('\t')
            ids.add(asin)
            image_paths[url].add(asin)
    return ids, image_paths
