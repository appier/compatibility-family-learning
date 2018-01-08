import argparse
import os

from queue import Queue
from threading import Thread

import numpy as np

from keras.applications import InceptionV3
from keras.applications.inception_v3 import preprocess_input
from keras.layers import GlobalAveragePooling2D
from keras.layers import Input
from keras.models import Model
from scipy.misc import imread, imresize
from tqdm import trange


def enqueue(queue, input_dir, batch_size):
    images = [os.path.join(input_dir, name) for name in os.listdir(input_dir)]
    for i in trange(0, len(images), batch_size):
        batch_paths = images[i:i + batch_size]
        batch_images = [
            preprocess_input(
                imresize(imread(image_path), (299, 299, 3)).astype(np.float32))
            for image_path in batch_paths
        ]
        batch_images = np.array(batch_images)
        queue.put((batch_paths, batch_images))
    queue.put((None, None))


def save(queue, output_dir):
    while True:
        batch_paths, latents = queue.get()
        if batch_paths is None:
            break
        for path, latent in zip(batch_paths, latents):
            output_path = os.path.join(output_dir, path.split('/')[-1])
            with open(output_path, 'wb') as outfile:
                np.savez_compressed(outfile, data=latent)


def extract(input_dir, output_dir, batch_size):
    queue = Queue(maxsize=100)
    inputs = Input(shape=(299, 299, 3), name='input_image')
    v3 = InceptionV3(include_top=False, weights='imagenet')
    encoded = v3(inputs)
    outputs = GlobalAveragePooling2D()(encoded)
    model = Model(inputs=inputs, outputs=outputs)

    worker = Thread(target=enqueue, args=(queue, input_dir, batch_size))
    worker.daemon = True
    worker.start()

    save_queue = Queue(maxsize=100)
    worker_save = Thread(target=save, args=(save_queue, output_dir))
    worker_save.daemon = True
    worker_save.start()

    os.makedirs(output_dir, exist_ok=True)
    while True:
        batch_paths, batch_images = queue.get()
        if batch_paths is None:
            save_queue.put((None, None))
            break
        batch_latents = model.predict_on_batch(batch_images)
        save_queue.put((batch_paths, batch_latents))

    worker_save.join()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-dir', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--batch-size', type=int, default=10)
    return parser.parse_args()


def main():
    args = parse_args()
    extract(**vars(args))


if __name__ == '__main__':
    main()
