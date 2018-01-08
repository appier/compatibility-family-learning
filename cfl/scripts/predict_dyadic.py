import argparse
import json
import logging
import os

import caffe
import numpy as np

from tqdm import tqdm

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-file', required=True)
    parser.add_argument('--weight-file', required=True)
    parser.add_argument('--id-path', required=True)
    parser.add_argument('--input-dir', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--batch-size', type=int, default=128)

    return parser.parse_args()


def main():
    args = parse_args()
    predict(**vars(args))


def predict(model_file, weight_file, id_path, input_dir, output_dir,
            batch_size):
    logger.warning('loading model..')
    caffe.set_mode_gpu()
    caffe.set_device(0)
    net = caffe.Classifier(
        model_file,
        weight_file,
        channel_swap=(2, 1, 0),
        mean=np.array([104.0, 117.0, 123.0]),
        raw_scale=255,
        image_dims=(256, 256))
    logger.warning('successfully loaded classifier')

    logger.warning('start converting images...')
    with open(id_path) as infile:
        ids = sorted(set(json.load(infile)))
    logger.warning('conevrt %d images', len(ids))

    total = set()
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    t = tqdm(range(0, len(ids), batch_size))
    for i in t:
        batch = [
            name for name in ids[i:i + batch_size]
            if os.path.exists(os.path.join(input_dir, name + '.jpg'))
        ]

        input_images = [
            caffe.io.load_image(os.path.join(input_dir, name + '.jpg'))
            for name in batch
        ]
        preds = net.predict(input_images, False)
        for name, pred in zip(batch, preds):
            path = os.path.join(output_dir, name + '.npy')
            with open(path, 'wb') as outfile:
                np.save(outfile, pred)
            total.add(name)
        t.set_postfix(total=len(total))


if __name__ == '__main__':
    main()
