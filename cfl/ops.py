import functools
import logging

import numpy as np
import tensorflow as tf

logger = logging.getLogger(__name__)


def lrelu(x, leak=0.2, name='lrelu'):
    with tf.name_scope(name):
        return tf.nn.relu(x) - leak * tf.nn.relu(-x)


def build_placeholder(shape, name, default=None, dtype=tf.float32):
    if default is None:
        return tf.placeholder(dtype, shape, name=name)
    else:
        return tf.placeholder_with_default(default, shape, name=name)


def compose(*functions):
    return functools.reduce(lambda f, g: lambda x: f(g(x)), functions,
                            lambda x: x)


def identity(tensor):
    return tensor


def reshapper(size):
    def fn(tensor):
        return tf.reshape(tensor, (-1, ) + tuple(size))

    return fn


def cropper(size):
    def fn(tensor):
        return tf.image.resize_image_with_crop_or_pad(tensor, size[0], size[1])

    return fn


def random_flip(tensor):
    return tf.map_fn(tf.image.random_flip_left_right, tensor)


def random_cropper(size):
    def fn(tensor):
        def random_crop_fn(value):
            return tf.random_crop(value, size)

        return tf.map_fn(random_crop_fn, tensor)

    return fn


def resizer(size):
    def fn(tensor):
        return tf.image.resize_images(tensor, size[:2])

    return fn


def normalize_v2(tensor,
                 input_shape,
                 scale=None,
                 mean=None,
                 norm=None,
                 clip_value_min=None,
                 clip_value_max=None):
    # always convert to tuple
    if mean is not None and not isinstance(mean, (list, tuple)):
        mean = (mean, )
    if norm is not None and not isinstance(norm, (list, tuple)):
        norm = (norm, )

    if scale is not None and scale != 1.:
        tensor *= scale

    # check if need to separate
    if ((mean and len(mean) > 1) or (norm and len(norm) > 1)):

        channels = tf.unstack(tensor, axis=-1)

        if mean:
            if len(mean) == 1:
                mean *= 3
            assert len(mean) == 3
            for i, mean_val in enumerate(mean):
                if mean_val != 0.:
                    channels[i] -= mean_val

        if norm:
            if len(norm) == 1:
                norm *= 3
            assert len(norm) == 3
            for i, norm_val in enumerate(norm):
                if norm_val != 1.:
                    channels[i] /= norm_val
        tensor = tf.stack(channels, axis=-1)

    else:
        # no channel swap and all are single values
        if mean and mean[0] != 0.:
            tensor -= mean[0]
        if norm and norm[0] != 1.:
            tensor /= norm[0]

    if clip_value_min is not None and clip_value_max is None:
        tensor = tf.maximum(clip_value_min, tensor)
    elif clip_value_min is None and clip_value_max is not None:
        tensor = tf.minimum(clip_value_max, tensor)
    elif clip_value_min is not None and clip_value_max is not None:
        tensor = tf.clip_by_value(tensor, clip_value_min, clip_value_max)

    if input_shape:
        input_size = 1
        for dim in input_shape:
            input_size *= dim
        tensor = tf.reshape(tensor, (-1, input_size))

    return tensor


def normalizer_v2(input_shape,
                  scale=None,
                  mean=None,
                  norm=None,
                  clip_value_min=None,
                  clip_value_max=None):
    def fn(tensor):
        return normalize_v2(
            tensor,
            input_shape,
            scale=scale,
            mean=mean,
            norm=norm,
            clip_value_min=clip_value_min,
            clip_value_max=clip_value_max)

    return fn


def unnormalize_v2(tensor, input_shape, scale=None, mean=None, norm=None):
    # always convert to tuple
    if mean is not None and not isinstance(mean, (list, tuple)):
        mean = (mean, )
    if norm is not None and not isinstance(norm, (list, tuple)):
        norm = (norm, )

    tensor = tf.reshape(tensor, (-1, ) + tuple(input_shape))

    # check if need to separate
    if ((mean and len(mean) > 1) or (norm and len(norm) > 1)):
        channels = tf.unstack(tensor, axis=-1)

        if norm:
            if len(norm) == 1:
                norm *= 3
            assert len(norm) == 3
            for i, norm_val in enumerate(norm):
                if norm_val != 1.:
                    channels[i] *= norm_val

        if mean:
            if len(mean) == 1:
                mean *= 3
            assert len(mean) == 3
            for i, mean_val in enumerate(mean):
                if mean_val != 0.:
                    channels[i] += mean_val

        tensor = tf.stack(channels, axis=-1)

    else:
        # no channel swap and all are single values
        if norm and norm[0] != 1.:
            tensor *= norm[0]
        if mean and mean[0] != 0.:
            tensor += mean[0]

    if scale is not None and scale != 1.:
        tensor /= scale

    return tensor


def unnormalizer_v2(input_shape, scale=None, mean=None, norm=None):
    def fn(tensor):
        return unnormalize_v2(
            tensor, input_shape, scale=scale, mean=mean, norm=norm)

    return fn


def normalize(tensor, scale, shift, clip_value_min=None, clip_value_max=None):
    tensor = tensor / scale + shift
    if clip_value_min is not None or clip_value_max is not None:
        tensor = tf.clip_by_value(tensor, clip_value_min, clip_value_max)
    return tensor


def normalizer(scale, shift, clip_value_min=None, clip_value_max=None):
    def fn(tensor):
        return normalize(
            tensor,
            scale,
            shift,
            clip_value_min=clip_value_min,
            clip_value_max=clip_value_max)

    return fn


def unnormalize(tensor, scale, shift):
    return (tensor - shift) * scale


def unnormalizer(scale, shift):
    def fn(tensor):
        return unnormalize(tensor, scale, shift)

    return fn


def arrange_grid(rows, num_rows, num_cols, image_shape, transpose=False):
    rows = tf.reshape(rows, (num_rows, num_cols, image_shape[0],
                             image_shape[1], image_shape[2]))
    if transpose:
        num_rows, num_cols = num_cols, num_rows
        rows = tf.transpose(rows, (1, 0, 2, 3, 4))

    height = image_shape[0] * num_rows
    width = image_shape[1] * num_cols

    rows = tf.transpose(rows, (0, 1, 3, 2, 4))
    rows = tf.reshape(rows, (num_rows, width, image_shape[0], image_shape[2]))
    rows = tf.transpose(rows, (0, 2, 1, 3))
    rows = tf.reshape(rows, (1, height, width, image_shape[2]))
    return rows


def np_arrange_grid(rows, num_rows, num_cols, image_shape, transpose=False):
    rows = np.reshape(rows, (num_rows, num_cols, image_shape[0],
                             image_shape[1], image_shape[2]))
    if transpose:
        num_rows, num_cols = num_cols, num_rows
        rows = np.transpose(rows, (1, 0, 2, 3, 4))

    height = image_shape[0] * num_rows
    width = image_shape[1] * num_cols

    rows = np.transpose(rows, (0, 1, 3, 2, 4))
    rows = np.reshape(rows, (num_rows, width, image_shape[0], image_shape[2]))
    rows = np.transpose(rows, (0, 2, 1, 3))
    rows = np.reshape(rows, (1, height, width, image_shape[2]))
    return rows


def dist_transformer(source_shape, input_shape, data_random_crop, data_mirror):
    if source_shape is not None and tuple(input_shape) != tuple(source_shape):
        if source_shape[0] > input_shape[0]:
            logger.info('crop image from %r to %r, random = %r', source_shape,
                        input_shape, data_random_crop)
            if data_random_crop:
                train_transformer = random_cropper(input_shape)
            else:
                train_transformer = cropper(input_shape)
            val_transformer = cropper(input_shape)
        else:
            logger.info('resize image from %r to %r', source_shape,
                        input_shape)
            train_transformer = resizer(input_shape[:2])
            val_transformer = resizer(input_shape[:2])

        train_transformer = compose(train_transformer, reshapper(source_shape))
        val_transformer = compose(val_transformer, reshapper(source_shape))

    else:
        train_transformer = reshapper(input_shape)
        val_transformer = reshapper(input_shape)

    if data_mirror:
        logger.info('mirrorring data.')
        train_transformer = compose(random_flip, train_transformer)

    return train_transformer, val_transformer


def dist_ae_transformer(input_shape, ae_shape):
    if ae_shape is not None and tuple(input_shape) != tuple(ae_shape):
        logger.info('resize image from %r to %r for autoencoder', input_shape,
                    ae_shape)
        return resizer(ae_shape[:2])

    else:
        return identity


def dist_normalizer(input_shape, ae_shape, data_scale, data_mean, data_norm,
                    latent_norm, data_type):
    clip_values = {
        'sigmoid': (0., 1.),
        'tanh': (-1., 1.),
        'relu': (0., None),
        'linear': (None, None)
    }

    clip_value_min, clip_value_max = clip_values[data_type]

    data_normalizer = normalizer_v2(
        input_shape,
        scale=data_scale,
        mean=data_mean,
        norm=data_norm,
        clip_value_min=clip_value_min,
        clip_value_max=clip_value_max)
    if latent_norm:
        latent_normalizer = normalizer_v2(
            None,
            scale=None,
            mean=None,
            norm=latent_norm,
            clip_value_min=None,
            clip_value_max=None)
    else:
        latent_normalizer = None

    data_unnormalizer = unnormalizer_v2(
        input_shape, scale=data_scale, mean=data_mean, norm=data_norm)

    if ae_shape is not None and tuple(ae_shape) != tuple(input_shape):
        ae_normalizer = normalizer_v2(
            ae_shape,
            scale=data_scale,
            mean=data_mean,
            norm=data_norm,
            clip_value_min=clip_value_min,
            clip_value_max=clip_value_max)

        ae_unnormalizer = unnormalizer_v2(
            ae_shape, scale=data_scale, mean=data_mean, norm=data_norm)
    else:
        ae_normalizer = data_normalizer
        ae_unnormalizer = data_unnormalizer

    return data_normalizer, data_unnormalizer, ae_normalizer, ae_unnormalizer, latent_normalizer
