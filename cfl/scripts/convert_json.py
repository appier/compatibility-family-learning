import argparse
import ast
import gzip
import json

from tqdm import tqdm


def parse(inpath, outpath):
    """parse
    convert to real json file
    """
    with gzip.open(inpath, 'r') as infile, gzip.open(outpath, 'w') as outfile:
        for line in tqdm(infile):
            item = ast.literal_eval(line.decode('utf8'))
            outfile.write(json.dumps(item).encode('utf8') + b'\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('inpath')
    parser.add_argument('outpath')
    args = parser.parse_args()

    parse(**vars(args))


if __name__ == '__main__':
    main()
