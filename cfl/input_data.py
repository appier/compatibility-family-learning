"""Input data utility
"""
import logging
import os
import struct

from argparse import Namespace
from array import array
from io import BytesIO
from collections import defaultdict

import numpy as np
import tensorflow as tf

from tqdm import trange
from numpy.random import RandomState
from scipy.misc import imread, imsave
from tensorflow.examples.tutorials import mnist

logger = logging.getLogger(__name__)


def dump_array(outfile, floats):
    """dump_array
    Dump array to binary file

    :param outfile: the output file
    :param floats: the array to dumpy
    """
    float_array = array('f', floats)
    float_array.tofile(outfile)


def dump_image(outfile, image, image_format='png'):
    """dump_image
    Dump image to binary file

    :param outfile: the output file
    :param image: the image array to dumpy
    """

    image_file = BytesIO()
    imsave(image_file, image, image_format)
    size = image_file.getbuffer().nbytes

    outfile.write(struct.pack('<i', size))
    outfile.write(image_file.getvalue())


def load_images(path):
    """load_images
    Load images from binary file

    :param path:
    """
    with open(path, 'rb') as infile:
        while True:
            asin = infile.read(10).decode('ascii')
            if not asin:
                break
            size = struct.unpack('<i', infile.read(4))[0]
            with BytesIO(infile.read(size)) as image_file:
                image = imread(image_file).astype(np.float32) / 255.
            yield asin, image.reshape((-1, ))


def load_images_by_offsets(path, offsets):
    """load_images_by_offsets
    Load images from binary file by its ofsets

    :param path:
    :param offsets:
    """
    images = []
    with open(path, 'rb') as infile:
        for offset in offsets:
            file_pos = offset + 10
            infile.seek(file_pos)
            size = struct.unpack('<i', infile.read(4))[0]

            with BytesIO(infile.read(size)) as image_file:
                image = imread(image_file).astype(np.float32) / 255.
            images.append(image)
    return np.array(images).reshape((len(offsets), -1))


def yield_double_images(path):
    """yield_double_images
    Yield asin, image pairs from double dataset

    :param path:
    """
    with open(path, 'rb') as infile:
        while True:
            asin = infile.read(10).decode('ascii')
            if not asin:
                break
            size1 = struct.unpack('<i', infile.read(4))[0]
            size2 = struct.unpack('<i', infile.read(4))[0]

            with BytesIO(infile.read(size1)) as image_file:
                image = imread(image_file)
                yield asin, image
            infile.seek(size2, 1)


def load_double_images_by_offsets(path, offsets, raw_latent=False):
    """load_double_images_by_offsets
    Load images and latents from binary file by its ofsets

    :param path:
    :param offsets:
    """
    images = []
    latents = []
    with open(path, 'rb') as infile:
        for offset in offsets:
            file_pos = offset + 10
            infile.seek(file_pos)
            size1 = struct.unpack('<i', infile.read(4))[0]
            size2 = struct.unpack('<i', infile.read(4))[0]

            with BytesIO(infile.read(size1)) as image_file:
                image = imread(image_file).astype(np.float32) / 255.
            images.append(image)
            if raw_latent:
                latent = array('f')
                latent.fromfile(infile, size2 // 4)
            else:
                with BytesIO(infile.read(size2)) as latent_file:
                    latent = np.load(latent_file)['data']
            latents.append(latent)
    return np.array(images).reshape(
        (len(offsets), -1)), np.array(latents).reshape((len(offsets), -1))


def load_asins_by_offsets(path, offsets):
    """load_asins_by_offsets
    Load asins from binary file by its offsets

    :param path:
    :param offsets:
    """
    asins = []
    with open(path, 'rb') as infile:
        for offset in offsets:
            infile.seek(offset)
            asins.append(infile.read(10).decode('ascii'))
    return asins


def load_double_offsets(path):
    """load_images_offsets
    Load the offsets of images from binary file

    :param path:
    """
    offset = 0
    offsets = {}
    with open(path, 'rb') as infile:
        while True:
            asin = infile.read(10).decode('ascii')
            if not asin:
                break
            offsets[asin] = offset

            size1 = struct.unpack('<i', infile.read(4))[0]
            size2 = struct.unpack('<i', infile.read(4))[0]
            offset += 10 + 4 + 4 + size1 + size2
            infile.seek(offset)
    return offsets


def load_images_offsets(path):
    """load_images_offsets
    Load the offsets of images from binary file

    :param path:
    """
    offset = 0
    offsets = {}
    with open(path, 'rb') as infile:
        while True:
            asin = infile.read(10).decode('ascii')
            if not asin:
                break
            offsets[asin] = offset

            size = struct.unpack('<i', infile.read(4))[0]
            offset += 10 + 4 + size
            infile.seek(offset)
    return offsets


def load_features(path, input_size=28 * 28):
    """load_features
    Load features from binary file

    :param path:
    :param input_size:
    """
    with open(path, 'rb') as infile:
        while True:
            asin = infile.read(10).decode('ascii')
            if not asin:
                break
            feature = array('f')
            feature.fromfile(infile, input_size)
            yield asin, feature


def load_features_by_positions(path, positions, input_size=28 * 28):
    """load_features_by_positions
    Load features from binary file by its positions

    :param path:
    :param positions:
    :param input_size:
    """
    features = []
    with open(path, 'rb') as infile:
        for pos in positions:
            file_pos = 10 + (input_size * 4 + 10) * pos
            infile.seek(file_pos)
            feature = array('f')
            feature.fromfile(infile, input_size)
            features.append(feature)
    return np.array(features)


def load_asins_by_positions(path, positions, input_size=28 * 28):
    """load_asins_by_positions
    Load asins from binary file by its positions

    :param path:
    :param positions:
    :param input_size:
    """
    asins = []
    with open(path, 'rb') as infile:
        for pos in positions:
            file_pos = (input_size * 4 + 10) * pos
            infile.seek(file_pos)
            asins.append(infile.read(10).decode('ascii'))
    return asins


def load_features_indices(path, input_size=28 * 28):
    """load_features_indices
    Load the indices of features from binary file

    :param path:
    :param input_size:
    """
    idx = 0
    indices = {}
    with open(path, 'rb') as infile:
        while True:
            asin = infile.read(10).decode('ascii')
            if not asin:
                break
            indices[asin] = idx
            idx += 1
            infile.seek(4 * input_size, 1)
    return indices


def load_meta_lines(path):
    """load_meta_lines
    Load meta lines from meta text file

    :param path:
    """
    with open(path) as infile:
        lines = []
        current_id = None
        for line in infile:
            if not line.startswith(' '):
                if current_id:
                    yield current_id, lines
                current_id = line.split(' ', 1)[0].strip()
                lines = [line]
            else:
                assert lines and current_id, 'must have valid id'
                lines.append(line)
        if current_id:
            yield current_id, lines


def load_data_sets(path,
                   input_size,
                   data_switch=False,
                   raw_latent=False,
                   is_image=False,
                   is_double=False,
                   directed=False,
                   reorder=False,
                   seed=633):
    """load_data_sets
    Load train / val / test splits

    :param path:
    :param input_size:
    :param data_switch:
    :param raw_latent:
    :param is_image:
    :param is_double:
    :param directed:
    :param reorder: Make test set labels ordered for visualization
    :param seed:
    """
    if is_image:
        logger.info('load image data...')
    data_train = SemiDataSet(
        path=os.path.join(path, 'train'),
        input_size=input_size,
        raw_latent=raw_latent,
        is_image=is_image,
        is_double=is_double,
        directed=directed,
        data_switch=data_switch,
        seed=seed)
    data_val = SemiDataSet(
        path=os.path.join(path, 'val'),
        input_size=input_size,
        is_image=is_image,
        is_double=is_double,
        raw_latent=raw_latent,
        directed=directed,
        seed=seed)
    data_test = SemiDataSet(
        path=os.path.join(path, 'test'),
        input_size=input_size,
        is_image=is_image,
        is_double=is_double,
        directed=directed,
        raw_latent=raw_latent,
        reorder=reorder,
        seed=seed)
    data = Namespace(train=data_train, val=data_val, test=data_test)
    return data


class SemiDataSet(object):
    """SemiDataSet"""

    def __init__(self,
                 path,
                 input_size=28 * 28,
                 data_switch=False,
                 is_image=False,
                 is_double=False,
                 directed=False,
                 reorder=False,
                 raw_latent=False,
                 seed=633):
        """__init__

        :param path:
        :param input_size:
        :param data_switch: random swap pairs
        :param is_image:
        :param is_double:
        :param directed:
        :param reorder: make labels ordered
        :param raw_latent:
        :param seed:
        """
        self._rng = RandomState(seed)
        self.input_size = input_size
        self.feature_path = os.path.join(path, 'features.b')
        self.is_image = is_image
        self.is_double = is_double
        self.directed = directed
        self.data_switch = data_switch
        self.raw_latent = raw_latent

        # unlabeled
        if is_image:
            if is_double:
                self.asins_to_index = load_double_offsets(self.feature_path)
            else:
                self.asins_to_index = load_images_offsets(self.feature_path)
            self.index_to_asins = {
                index: asin
                for asin, index in self.asins_to_index.items()
            }
            self.num_examples = len(self.index_to_asins)
            self.item_indices = np.array(sorted(list(self.index_to_asins)))
            perm = self._rng.permutation(self.num_examples)
            self.item_indices = self.item_indices[perm]
        else:
            self.asins_to_index = load_features_indices(
                self.feature_path, input_size=self.input_size)
            self.index_to_asins = {
                index: asin
                for asin, index in self.asins_to_index.items()
            }
            self.num_examples = max(self.index_to_asins)
            self.item_indices = np.arange(self.num_examples)

        self.head_unlabeled = 0

        if self.directed:
            source_ids = []
            with open(os.path.join(path, 'source.txt')) as infile:
                for line in infile:
                    asin = line.strip()
                    source_ids.append(asin)
            self.source_indices = np.array(
                sorted([self.asins_to_index[asin] for asin in source_ids]))
            self.num_source = self.source_indices.shape[0]
            perm = self._rng.permutation(self.num_source)
            self.source_indices = self.source_indices[perm]
            self.head_source = 0

            target_ids = []
            with open(os.path.join(path, 'target.txt')) as infile:
                for line in infile:
                    asin = line.strip()
                    target_ids.append(asin)
            self.target_indices = np.array(
                sorted([self.asins_to_index[asin] for asin in target_ids]))
            self.num_target = self.target_indices.shape[0]
            perm = self._rng.permutation(self.num_target)
            self.target_indices = self.target_indices[perm]
            self.head_target = 0

        # pairs
        self.pairs_pos = []
        self.pairs_neg = []

        with open(os.path.join(path, 'pairs_pos.txt')) as infile:
            if reorder:
                buckets = defaultdict(list)
                for line in infile:
                    a, _, b = line.strip().split()
                    buckets[a[0]].append((a, b))

                lines = []
                while sum(len(v) for v in buckets.values()) > 0:
                    for _, v in sorted(buckets.items(), key=lambda x: x[0]):
                        if len(v) > 0:
                            lines.append(v.pop())
                for a, b in lines:
                    self.pairs_pos.append(
                        [self.asins_to_index[a], self.asins_to_index[b]])
            else:
                for line in infile:
                    a, _, b = line.strip().split()
                    self.pairs_pos.append(
                        [self.asins_to_index[a], self.asins_to_index[b]])

        with open(os.path.join(path, 'pairs_neg.txt')) as infile:
            for line in infile:
                a, _, b = line.strip().split()
                self.pairs_neg.append(
                    [self.asins_to_index[a], self.asins_to_index[b]])

        self.pairs_pos = np.array(self.pairs_pos)
        self.pairs_neg = np.array(self.pairs_neg)
        self.head_labeled_pos = 0
        self.head_labeled_neg = 0
        self.num_examples_labeled_pos = self.pairs_pos.shape[0]
        self.num_examples_labeled_neg = self.pairs_neg.shape[0]

    def next_batch(self, batch_size, return_labels=False):
        return self.next_labeled_batch(batch_size, return_labels)

    def _load_features_by_positions(self, indices):
        if self.is_image:
            if self.is_double:
                return load_double_images_by_offsets(
                    self.feature_path, indices, raw_latent=self.raw_latent)
            else:
                return load_images_by_offsets(self.feature_path, indices)
        else:
            return load_features_by_positions(
                self.feature_path, indices, input_size=self.input_size)

    def _load_asins_by_positions(self, indices):
        if self.is_image:
            return load_asins_by_offsets(self.feature_path, indices)
        else:
            return load_asins_by_positions(
                self.feature_path, indices, input_size=self.input_size)

    def whole_unlabeled_batches(self, batch_size, source_ids=False):
        for i in trange(0, self.num_examples, batch_size):
            positions = self.item_indices[i:i + batch_size]
            data = self._load_features_by_positions(positions)
            if self.is_double:
                data = list(data)
            else:
                data = [data]

            if source_ids:
                asins = self._load_asins_by_positions(positions)
                data += [asins]

            yield data

    def whole_pos_batches(self, batch_size, source_ids=False):
        for i in trange(0, self.num_examples_labeled_pos, batch_size):
            src_pos = self._load_features_by_positions(
                self.pairs_pos[i:i + batch_size, 0])
            dst_pos = self._load_features_by_positions(
                self.pairs_pos[i:i + batch_size, 1])
            ids = self._load_asins_by_positions(
                self.pairs_pos[i:i + batch_size, 0])
            if source_ids:
                if self.is_double:
                    yield src_pos[0], src_pos[1], dst_pos[0], dst_pos[1], ids
                else:
                    yield src_pos, dst_pos, ids
            else:
                if self.is_double:
                    yield src_pos[0], src_pos[1], dst_pos[0], dst_pos[1]
                else:
                    yield src_pos, dst_pos

    def whole_neg_batches(self, batch_size, source_ids=False):
        for i in trange(0, self.num_examples_labeled_neg, batch_size):
            src_neg = self._load_features_by_positions(
                self.pairs_neg[i:i + batch_size, 0])
            dst_neg = self._load_features_by_positions(
                self.pairs_neg[i:i + batch_size, 1])

            ids = self._load_asins_by_positions(
                self.pairs_neg[i:i + batch_size, 0])
            if source_ids:
                if self.is_double:
                    yield src_neg[0], src_neg[1], dst_neg[0], dst_neg[1], ids
                else:
                    yield src_neg, dst_neg, ids
            else:
                if self.is_double:
                    yield src_neg[0], src_neg[1], dst_neg[0], dst_neg[1]
                else:
                    yield src_neg, dst_neg

    def next_labeled_batch(self, batch_size, return_labels=False):
        if self.head_labeled_pos + batch_size > self.num_examples_labeled_pos:
            self.head_labeled_pos = 0
            perm = self._rng.permutation(self.num_examples_labeled_pos)
            self.pairs_pos = self.pairs_pos[perm]

        if self.head_labeled_neg + batch_size > self.num_examples_labeled_neg:
            self.head_labeled_neg = 0
            perm = self._rng.permutation(self.num_examples_labeled_neg)
            self.pairs_neg = self.pairs_neg[perm]

        positions_pos = self.pairs_pos[self.head_labeled_pos:
                                       self.head_labeled_pos + batch_size]
        positions_neg = self.pairs_neg[self.head_labeled_neg:
                                       self.head_labeled_neg + batch_size]

        if batch_size > self.num_examples_labeled_pos:
            indices = self._rng.choice(self.num_examples_labeled_pos,
                                       batch_size)
            positions_pos = self.pairs_pos[indices]

        if batch_size > self.num_examples_labeled_neg:
            indices = self._rng.choice(self.num_examples_labeled_neg,
                                       batch_size)
            positions_neg = self.pairs_neg[indices]
        assert positions_pos.shape[0] == batch_size
        assert positions_neg.shape[0] == batch_size

        src_pos = self._load_features_by_positions(positions_pos[:, 0])
        dst_pos = self._load_features_by_positions(positions_pos[:, 1])
        src_neg = self._load_features_by_positions(positions_neg[:, 0])
        dst_neg = self._load_features_by_positions(positions_neg[:, 1])

        if self.data_switch and self._rng.rand() > 0.5:
            src_pos, dst_pos = dst_pos, src_pos
            src_neg, dst_neg = dst_neg, src_neg

        self.head_labeled_pos += batch_size
        self.head_labeled_neg += batch_size

        if return_labels:
            raise NotImplementedError()
        else:
            if self.is_double:
                return src_pos[0], src_pos[1], dst_pos[0], dst_pos[1], src_neg[
                    0], src_neg[1], dst_neg[0], dst_neg[1]
            else:
                return src_pos, dst_pos, src_neg, dst_neg

    def next_unlabeled_batch(self,
                             batch_size,
                             return_labels=False,
                             source_ids=False):
        if self.head_unlabeled + batch_size > self.num_examples:
            self.head_unlabeled = 0
            perm = self._rng.permutation(self.num_examples)
            self.item_indices = self.item_indices[perm]

        positions = self.item_indices[self.head_unlabeled:self.head_unlabeled +
                                      batch_size]
        data = self._load_features_by_positions(positions)
        self.head_unlabeled += batch_size

        if self.is_double:
            data = list(data)
        else:
            data = [data]

        if return_labels or source_ids:
            asins = self._load_asins_by_positions(positions)
            if return_labels:
                labels = [self.categories[asin] for asin in asins]
                labels = np.array(labels)

                data += [labels]
            if source_ids:
                data += [asins]
        return data

    def next_source_batch(self,
                          batch_size,
                          return_labels=False,
                          source_ids=False):
        if not self.directed:
            return self.next_unlabeled_batch(
                batch_size, return_labels=return_labels, source_ids=source_ids)

        if self.head_source + batch_size > self.num_source:
            self.head_source = 0
            perm = self._rng.permutation(self.num_source)
            self.source_indices = self.source_indices[perm]

        positions = self.source_indices[self.head_source:self.head_source +
                                        batch_size]
        if batch_size > self.num_source:
            indices = self._rng.choice(self.num_source, batch_size)
            positions = self.source_indices[indices]
        assert positions.shape[0] == batch_size
        data = self._load_features_by_positions(positions)
        self.head_source += batch_size
        if self.is_double:
            data = list(data)
        else:
            data = [data]

        if return_labels or source_ids:
            asins = self._load_asins_by_positions(positions)
            if return_labels:
                labels = [self.categories[asin] for asin in asins]
                labels = np.array(labels)

                data += [labels]
            if source_ids:
                data += [asins]

        return data

    def next_target_batch(self,
                          batch_size,
                          return_labels=False,
                          source_ids=False):
        if not self.directed:
            return self.next_unlabeled_batch(
                batch_size, return_labels=return_labels, source_ids=source_ids)

        if self.head_target + batch_size > self.num_target:
            self.head_target = 0
            perm = self._rng.permutation(self.num_target)
            self.target_indices = self.target_indices[perm]

        positions = self.target_indices[self.head_target:self.head_target +
                                        batch_size]
        data = self._load_features_by_positions(positions)
        self.head_target += batch_size
        if self.is_double:
            data = list(data)
        else:
            data = [data]

        if return_labels or source_ids:
            asins = self._load_asins_by_positions(positions)
            if return_labels:
                labels = [self.categories[asin] for asin in asins]
                labels = np.array(labels)

                data += [labels]
            if source_ids:
                data += [asins]
        return data


class SemiMNIST(object):
    def __init__(self,
                 data_type='diffone',
                 path='data/MNIST_data',
                 split='train',
                 all_random=False,
                 symmetric=False,
                 resample=False,
                 labeled_percent=0.5,
                 sample_rate=2,
                 seed=633):
        self._rng = RandomState(seed)
        # diffone: b-a % 5 = +- 1
        if resample:
            if split == 'test':
                self.mnist = mnist.input_data.read_data_sets(
                    path, one_hot=False, dtype=tf.uint8).test
            elif split in {'train', 'validation'}:
                self.mnist = mnist.input_data.read_data_sets(
                    path, one_hot=False, dtype=tf.uint8,
                    validation_size=0).train
                perm = self._rng.permutation(self.mnist._labels.shape[0])
                self.mnist._labels = self.mnist.labels[perm]
                self.mnist._images = self.mnist.images[perm]
                if split == 'validation':
                    self.mnist._labels = self.mnist.labels[:5000]
                    self.mnist._images = self.mnist.images[:5000]
                else:
                    self.mnist._labels = self.mnist.labels[5000:]
                    self.mnist._images = self.mnist.images[5000:]
            else:
                raise NotImplementedError()
        else:
            if split == 'train':
                self.mnist = mnist.input_data.read_data_sets(
                    path, one_hot=False, dtype=tf.uint8).train
            elif split == 'validation':
                self.mnist = mnist.input_data.read_data_sets(
                    path, one_hot=False, dtype=tf.uint8).validation
            elif split == 'test':
                self.mnist = mnist.input_data.read_data_sets(
                    path, one_hot=False, dtype=tf.uint8).test
            else:
                raise NotImplementedError()

        self.mnist._images = self.mnist._images.astype(np.float32)
        self.mnist._labels = self.mnist._labels.astype(np.int32)
        self.data_type = data_type
        self.all_random = all_random
        self.symmetric = symmetric

        if data_type == 'diffone':
            self._init_diffone(
                labeled_percent=labeled_percent, sample_rate=sample_rate)
        elif data_type == 'diffone_all':
            self._init_diffone(
                labeled_percent=labeled_percent,
                sample_rate=sample_rate,
                all_digits=True,
                symmetric=symmetric)
        else:
            raise NotImplementedError()

    def next_labeled_batch(self, batch_size, return_labels=False):
        if self.data_type.startswith('diffone'):
            return self.next_labeled_batch_diffone(
                batch_size, return_labels=return_labels)
        else:
            raise NotImplementedError()

    def next_unlabeled_batch(self, batch_size):
        if self.data_type.startswith('diffone'):
            return self.next_unlabeled_batch_diffone(batch_size)
        else:
            raise NotImplementedError()

    def next_labeled_batch_diffone(self, batch_size, return_labels):
        if self.head_labeled + batch_size > self.num_examples_labeled:
            self.head_labeled = 0
            perm = self._rng.permutation(self.num_examples_labeled)
            self.pairs_pos = self.pairs_pos[perm]
            self.pairs_neg = self.pairs_neg[perm]

        images_pos = self.mnist._images[self.pairs_pos[
            self.head_labeled:self.head_labeled + batch_size]]
        images_neg = self.mnist._images[self.pairs_neg[
            self.head_labeled:self.head_labeled + batch_size]]
        labels_pos = self.mnist._labels[self.pairs_pos[
            self.head_labeled:self.head_labeled + batch_size]]
        labels_neg = self.mnist._labels[self.pairs_neg[
            self.head_labeled:self.head_labeled + batch_size]]
        self.head_labeled += batch_size

        if return_labels:
            return images_pos, labels_pos, images_neg, labels_neg
        else:
            return images_pos, images_neg

    def next_unlabeled_batch_diffone(self, batch_size):
        if self.head_unlabeled + batch_size > self.num_examples_unlabeled:
            self.head_unlabeled = 0
            perm = self._rng.permutation(self.num_examples_unlabeled)
            self.indices_unlabeled = self.indices_unlabeled[perm]

        images = self.mnist._images[self.indices_unlabeled[
            self.head_unlabeled:self.head_unlabeled + batch_size]]
        cates = self.cate_labels[self.indices_unlabeled[
            self.head_unlabeled:self.head_unlabeled + batch_size]]
        self.head_unlabeled += batch_size

        return images, cates

    def _init_diffone(self,
                      labeled_percent,
                      sample_rate,
                      all_digits=False,
                      symmetric=False):
        labels = self.mnist.labels
        num_examples = labels.shape[0]
        labeled_num = int(num_examples * labeled_percent)

        indices = self._rng.choice(num_examples, labeled_num, replace=False)
        # split to 10 sets
        data_splits = [indices[labels[indices] == i] for i in range(10)]
        # for first n sets sample positive & negative for each item

        base_sets = 10 if all_digits else 5
        pairs_pos = []
        pairs_neg = []

        for i in range(base_sets):
            base_indices = np.expand_dims(data_splits[i], 1)
            # sample positives
            if all_digits:
                # 0 -> 1 2 (+1 +2)
                # 1 -> 2 3
                # 9 -> 0 1
                if symmetric:
                    t1 = (i + 9) % 10
                    t2 = (i + 1) % 10
                else:
                    t1 = (i + 1) % 10
                    t2 = (i + 2) % 10
            else:
                # 0 -> 6 9  1 -1
                # 1 -> 5 7
                # 4 -> 8 5
                t1 = ((i + 1) % 5) + 5
                t2 = ((i + 4) % 5) + 5

            matched_indices = indices[(labels[indices] == t1) | (labels[
                indices] == t2)]
            for _ in range(sample_rate):
                m_indices = self._rng.choice(matched_indices,
                                             base_indices.shape[0])
                m_indices = np.expand_dims(m_indices, 1)
                pairs_pos.append(
                    np.concatenate([base_indices, m_indices], axis=1))

            # sample negatives
            unmatched_indices = indices[(labels[indices] != t1) &
                                        (labels[indices] != t2)]
            for _ in range(sample_rate):
                um_indices = self._rng.choice(unmatched_indices,
                                              base_indices.shape[0])
                um_indices = np.expand_dims(um_indices, 1)
                pairs_neg.append(
                    np.concatenate([base_indices, um_indices], axis=1))

        pairs_pos = np.concatenate(pairs_pos)
        pairs_neg = np.concatenate(pairs_neg)

        if self.all_random:
            total_size = pairs_pos.shape[0]
            size = total_size // 2
            logger.warning('all_random: reduce size %d -> %d', total_size,
                           size)
            pos_indices = np.concatenate(
                [np.ones(size), np.zeros(total_size - size)]).astype(np.bool)
            pos_indices = self._rng.permutation(pos_indices)
            pairs_pos = pairs_pos[pos_indices][:size]
            pairs_neg = pairs_neg[~pos_indices][:size]
            assert pairs_pos.shape[0] == pairs_neg.shape[0]

        perm = self._rng.permutation(pairs_pos.shape[0])

        self.pairs_pos = pairs_pos[perm]
        self.pairs_neg = pairs_neg[perm]
        self.num_examples_labeled = self.pairs_pos.shape[0]
        self.head_labeled = 0

        self.indices_labeled = indices
        self.indices_unlabeled = np.arange(num_examples)
        self.cate_labels = (self.mnist._labels >= 5).astype(np.int32)
        self.num_examples_unlabeled = num_examples
        self.head_unlabeled = 0
