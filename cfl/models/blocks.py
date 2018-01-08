import tensorflow as tf

from ..layers import conv2d_transpose_weight_norm
from ..layers import conv2d_weight_norm
from ..layers import conv2d_subpixel
from ..layers import fully_connected_weight_norm
from ..ops import lrelu
from ..utils import reduce_product
from .base import DistBase
from .base import ModelBase


class Thresholder(ModelBase):
    def __init__(self, X, name='Thresholder', reuse=False):
        with tf.variable_scope(name, reuse=reuse) as scope:
            super().__init__(scope)

            with tf.variable_scope('threshold'):
                self.raw_threshold = tf.get_variable(
                    'threshold', [], initializer=tf.constant_initializer(1e-6))
                self.threshold = tf.maximum(self.raw_threshold, 1e-6)
            self.outputs = -1.0 * X + self.threshold


class SRGenerator(ModelBase):
    def __init__(self,
                 zs,
                 input_shape,
                 batch_size,
                 t_dim=None,
                 dim=64,
                 initializer=tf.contrib.layers.xavier_initializer(),
                 regularizer=None,
                 activation_fn=tf.nn.sigmoid,
                 name='generator',
                 reuse=False):

        self.input_shape = input_shape
        self.input_size = reduce_product(input_shape)
        self.batch_size = batch_size
        self.regularizer = regularizer
        self.initializer = initializer

        start_dim = min(input_shape[0], input_shape[1])
        channels = input_shape[-1]

        nb_upconv = 0
        while start_dim % 2 == 0 and start_dim > 4:
            start_dim //= 2
            nb_upconv += 1

        with tf.variable_scope(name, reuse=reuse) as scope:
            super().__init__(scope)

            if isinstance(zs, list):
                if t_dim is not None:
                    with tf.variable_scope('fc_t'):
                        zs[-1] = fully_connected_weight_norm(
                            inputs=zs[-1],
                            num_outputs=t_dim,
                            activation_fn=lrelu,
                            weights_regularizer=regularizer,
                            weights_initializer=initializer,
                            biases_initializer=tf.zeros_initializer())
                outputs = tf.concat(zs, axis=-1)
            else:
                outputs = zs

            with tf.variable_scope('fc1'):
                outputs = fully_connected_weight_norm(
                    inputs=outputs,
                    num_outputs=dim * start_dim * start_dim,
                    activation_fn=tf.nn.relu,
                    weights_initializer=initializer,
                    weights_regularizer=regularizer,
                    biases_initializer=tf.zeros_initializer())
                reshape_shape = (-1, start_dim, start_dim, dim)
                outputs = tf.reshape(outputs, reshape_shape)

            for i in range(nb_upconv - 1):
                with tf.variable_scope('subpixel_block{}'.format(i + 1)):
                    nb_filters = 4 * dim * (2**(nb_upconv - i - 1))
                    outputs = conv2d_weight_norm(
                        inputs=outputs,
                        num_outputs=nb_filters,
                        kernel_size=(3, 3),
                        stride=(1, 1),
                        padding='SAME',
                        activation_fn=None,
                        weights_initializer=initializer,
                        weights_regularizer=regularizer)
                    outputs = conv2d_subpixel(
                        inputs=outputs, scale=2, activation_fn=tf.nn.relu)

            with tf.variable_scope('outputs'):
                outputs = conv2d_weight_norm(
                    inputs=outputs,
                    num_outputs=4 * channels,
                    kernel_size=(3, 3),
                    stride=(1, 1),
                    padding='SAME',
                    activation_fn=None,
                    weights_initializer=initializer,
                    weights_regularizer=regularizer)
                outputs = conv2d_subpixel(inputs=outputs, scale=2)
                outputs = tf.contrib.layers.flatten(outputs)
                self.outputs = outputs
                self.activations = activation_fn(
                    self.outputs) if activation_fn else self.outputs


class SRDiscriminator(ModelBase):
    def __init__(self,
                 X,
                 latent_size,
                 input_shape,
                 batch_size,
                 num_classes,
                 t=None,
                 t_dim=None,
                 dim=32,
                 max_dim=512,
                 min_dim=4,
                 initializer=tf.contrib.layers.xavier_initializer(),
                 regularizer=None,
                 disc_activation_fn=tf.nn.sigmoid,
                 cls_activation_fn=tf.nn.softmax,
                 latent_activation_fn=None,
                 name='discriminator',
                 reuse=False):
        self.input_shape = input_shape
        self.batch_size = batch_size
        self.regularizer = regularizer
        self.initializer = initializer

        with tf.variable_scope(name, reuse=reuse) as scope:
            super().__init__(scope)

            outputs = tf.reshape(X, (-1, ) + input_shape)

            with tf.variable_scope('conv'):
                outputs = conv2d_weight_norm(
                    inputs=outputs,
                    num_outputs=dim,
                    kernel_size=(4, 4),
                    stride=(2, 2),
                    padding='SAME',
                    activation_fn=lrelu,
                    weights_initializer=initializer,
                    weights_regularizer=regularizer)

            start_dim = min(input_shape[0], input_shape[1])
            nb_upconv = 0
            while start_dim % 2 == 0 and start_dim > min_dim:
                start_dim //= 2
                nb_upconv += 1

            # convolutions
            for i in range(nb_upconv):
                with tf.variable_scope('conv{}'.format(i + 1)):
                    for j in range(2):
                        res_outputs = conv2d_weight_norm(
                            inputs=outputs,
                            num_outputs=dim,
                            kernel_size=(3, 3),
                            stride=(1, 1),
                            padding='SAME',
                            activation_fn=lrelu,
                            weights_initializer=initializer,
                            weights_regularizer=regularizer)
                        res_outputs = conv2d_weight_norm(
                            inputs=res_outputs,
                            num_outputs=dim,
                            kernel_size=(3, 3),
                            stride=(1, 1),
                            padding='SAME',
                            activation_fn=None,
                            weights_initializer=initializer,
                            weights_regularizer=regularizer)
                        outputs = lrelu(res_outputs + outputs)

                    if t is not None and i == 3:
                        if t_dim is not None:
                            with tf.variable_scope('fc_t'):
                                t = fully_connected_weight_norm(
                                    inputs=t,
                                    num_outputs=t_dim,
                                    activation_fn=lrelu,
                                    weights_regularizer=regularizer,
                                    weights_initializer=initializer,
                                    biases_initializer=tf.zeros_initializer())
                        t = tf.expand_dims(t, 1)
                        t = tf.expand_dims(t, 2)
                        t = tf.tile(t, [1, start_dim, start_dim, 1])
                        outputs = tf.concat([outputs, t], axis=3)

                    dim *= 2
                    outputs = conv2d_weight_norm(
                        inputs=outputs,
                        num_outputs=dim,
                        kernel_size=(4, 4),
                        stride=(2, 2),
                        padding='SAME',
                        activation_fn=lrelu,
                        weights_initializer=initializer,
                        weights_regularizer=regularizer)

            flatten_outputs = tf.contrib.layers.flatten(outputs)
            with tf.variable_scope('disc_outputs'):
                disc_outputs = fully_connected_weight_norm(
                    inputs=flatten_outputs,
                    num_outputs=1,
                    activation_fn=None,
                    weights_regularizer=regularizer,
                    weights_initializer=initializer,
                    biases_initializer=tf.zeros_initializer())
                self.disc_outputs = disc_outputs
                self.disc_activations = disc_activation_fn(
                    self.
                    disc_outputs) if disc_activation_fn else self.disc_outputs

            if latent_size:
                with tf.variable_scope('latent_outputs'):
                    latent_outputs = fully_connected_weight_norm(
                        inputs=flatten_outputs,
                        num_outputs=latent_size,
                        activation_fn=None,
                        weights_regularizer=regularizer,
                        weights_initializer=initializer,
                        biases_initializer=tf.zeros_initializer())
                    self.latent_outputs = latent_outputs
                    self.latent_activations = latent_activation_fn(
                        self.latent_outputs
                    ) if latent_activation_fn else self.latent_outputs

            if num_classes:
                with tf.variable_scope('cls_outputs'):
                    cls_outputs = fully_connected_weight_norm(
                        inputs=flatten_outputs,
                        num_outputs=num_classes,
                        activation_fn=None,
                        weights_regularizer=regularizer,
                        weights_initializer=initializer)
                    self.cls_outputs = cls_outputs
                    self.cls_activations = cls_activation_fn(
                        self.
                        cls_outputs) if cls_activation_fn else self.cls_outputs


class ConvTransposeGenerator(ModelBase):
    def __init__(self,
                 zs,
                 input_shape,
                 batch_size,
                 t_dim=None,
                 dim=64,
                 initializer=tf.contrib.layers.xavier_initializer(),
                 regularizer=None,
                 activation_fn=tf.nn.sigmoid,
                 name='generator',
                 reuse=False):

        self.input_shape = input_shape
        self.input_size = reduce_product(input_shape)
        self.batch_size = batch_size
        self.regularizer = regularizer
        self.initializer = initializer

        start_dim = min(input_shape[0], input_shape[1])
        channels = input_shape[-1]

        nb_upconv = 0
        while start_dim % 2 == 0 and start_dim > 4:
            start_dim //= 2
            nb_upconv += 1
        scale = 2**(nb_upconv - 1)

        with tf.variable_scope(name, reuse=reuse) as scope:
            super().__init__(scope)

            if isinstance(zs, list):
                if t_dim is not None:
                    with tf.variable_scope('fc_t'):
                        zs[-1] = fully_connected_weight_norm(
                            inputs=zs[-1],
                            num_outputs=t_dim,
                            activation_fn=lrelu,
                            weights_regularizer=regularizer,
                            weights_initializer=initializer,
                            biases_initializer=tf.zeros_initializer())
                outputs = tf.concat(zs, axis=-1)
            else:
                outputs = zs

            with tf.variable_scope('fc1'):
                outputs = fully_connected_weight_norm(
                    inputs=outputs,
                    num_outputs=dim * scale * start_dim * start_dim,
                    activation_fn=tf.nn.relu,
                    weights_initializer=initializer,
                    weights_regularizer=regularizer,
                    biases_initializer=tf.zeros_initializer())
                reshape_shape = (-1, start_dim, start_dim, dim * scale)
                outputs = tf.reshape(outputs, reshape_shape)

            for i in range(nb_upconv - 1):
                with tf.variable_scope('conv_t{}'.format(i + 1)):
                    nb_filters = dim * (2**(nb_upconv - i - 1))
                    outputs = conv2d_transpose_weight_norm(
                        inputs=outputs,
                        num_outputs=nb_filters,
                        kernel_size=(5, 5),
                        stride=(2, 2),
                        padding='SAME',
                        activation_fn=tf.nn.relu,
                        weights_regularizer=regularizer,
                        weights_initializer=initializer)

            with tf.variable_scope('outputs'):
                outputs = conv2d_transpose_weight_norm(
                    inputs=outputs,
                    num_outputs=channels,
                    kernel_size=(5, 5),
                    stride=(2, 2),
                    padding='SAME',
                    activation_fn=None,
                    weights_regularizer=regularizer,
                    weights_initializer=initializer)
                outputs = tf.contrib.layers.flatten(outputs)
                self.outputs = outputs
                self.activations = activation_fn(
                    self.outputs) if activation_fn else self.outputs


class ConvDiscriminator(ModelBase):
    def __init__(self,
                 X,
                 latent_size,
                 input_shape,
                 batch_size,
                 num_classes,
                 t=None,
                 t_dim=None,
                 dim=64,
                 max_dim=512,
                 min_dim=4,
                 initializer=tf.contrib.layers.xavier_initializer(),
                 regularizer=None,
                 disc_activation_fn=tf.nn.sigmoid,
                 cls_activation_fn=tf.nn.softmax,
                 latent_activation_fn=None,
                 name='discriminator',
                 reuse=False):
        self.input_shape = input_shape
        self.batch_size = batch_size
        self.regularizer = regularizer
        self.initializer = initializer

        start_dim = min(input_shape[0], input_shape[1])
        nb_upconv = 0
        while start_dim % 2 == 0 and start_dim > min_dim:
            start_dim //= 2
            nb_upconv += 1

        with tf.variable_scope(name, reuse=reuse) as scope:
            super().__init__(scope)

            outputs = tf.reshape(X, (-1, ) + input_shape)

            # convolutions
            for i in range(nb_upconv):
                with tf.variable_scope('conv{}'.format(i + 1)):
                    outputs = conv2d_weight_norm(
                        inputs=outputs,
                        num_outputs=dim,
                        kernel_size=(5, 5),
                        stride=(2, 2),
                        padding='SAME',
                        activation_fn=lrelu,
                        weights_initializer=initializer,
                        weights_regularizer=regularizer)
                    if t is not None and i == nb_upconv - 2:
                        if t_dim is not None:
                            with tf.variable_scope('fc_t'):
                                t = fully_connected_weight_norm(
                                    inputs=t,
                                    num_outputs=t_dim,
                                    activation_fn=lrelu,
                                    weights_regularizer=regularizer,
                                    weights_initializer=initializer,
                                    biases_initializer=tf.zeros_initializer())
                        t = tf.expand_dims(t, 1)
                        t = tf.expand_dims(t, 2)
                        t = tf.tile(t, [1, start_dim * 2, start_dim * 2, 1])
                        outputs = tf.concat([outputs, t], axis=3)

                dim = min(dim * 2, max_dim)

            flatten_outputs = tf.contrib.layers.flatten(outputs)
            with tf.variable_scope('disc_outputs'):
                disc_outputs = fully_connected_weight_norm(
                    inputs=flatten_outputs,
                    num_outputs=1,
                    activation_fn=None,
                    weights_regularizer=regularizer,
                    weights_initializer=initializer,
                    biases_initializer=tf.zeros_initializer())
                self.disc_outputs = disc_outputs
                self.disc_activations = disc_activation_fn(
                    self.
                    disc_outputs) if disc_activation_fn else self.disc_outputs

            if latent_size:
                with tf.variable_scope('latent_outputs'):
                    latent_outputs = fully_connected_weight_norm(
                        inputs=flatten_outputs,
                        num_outputs=latent_size,
                        activation_fn=None,
                        weights_regularizer=regularizer,
                        weights_initializer=initializer,
                        biases_initializer=tf.zeros_initializer())
                    self.latent_outputs = latent_outputs
                    self.latent_activations = latent_activation_fn(
                        self.latent_outputs
                    ) if latent_activation_fn else self.latent_outputs

            if num_classes:
                with tf.variable_scope('cls_outputs'):
                    cls_outputs = fully_connected_weight_norm(
                        inputs=flatten_outputs,
                        num_outputs=num_classes,
                        activation_fn=None,
                        weights_regularizer=regularizer,
                        weights_initializer=initializer)
                    self.cls_outputs = cls_outputs
                    self.cls_activations = cls_activation_fn(
                        self.
                        cls_outputs) if cls_activation_fn else self.cls_outputs


class FCDiscriminator(ModelBase):
    def __init__(self,
                 inputs,
                 layers=[300, 300],
                 initializer=tf.contrib.layers.xavier_initializer(),
                 disc_activation_fn=tf.nn.sigmoid,
                 name='discriminator',
                 reuse=False):
        self.initializer = initializer

        with tf.variable_scope(name, reuse=reuse) as scope:
            super().__init__(scope)

            outputs = inputs
            # convolutions
            for i, num_outputs in enumerate(layers):
                with tf.variable_scope('fc{}'.format(i + 1)):
                    outputs = fully_connected_weight_norm(
                        inputs=outputs,
                        num_outputs=num_outputs,
                        activation_fn=lrelu,
                        weights_initializer=initializer)

            with tf.variable_scope('disc_outputs'):
                disc_outputs = fully_connected_weight_norm(
                    inputs=outputs,
                    num_outputs=1,
                    activation_fn=None,
                    weights_initializer=initializer,
                    biases_initializer=tf.zeros_initializer())
                self.disc_outputs = disc_outputs
                self.disc_activations = disc_activation_fn(
                    self.
                    disc_outputs) if disc_activation_fn else self.disc_outputs


class FCPCD(DistBase):
    """FCPCD

    Encode vectors to a latent space and
    produce `num_components` prototype vectors.
    """

    def __init__(self,
                 X,
                 num_outputs,
                 num_components,
                 input_shape,
                 batch_size,
                 gate=None,
                 dist_type='pcd',
                 layer_sizes=None,
                 initializer=tf.contrib.layers.xavier_initializer(),
                 regularizer=None,
                 layer_activation_fn=lrelu,
                 activation_fn=None,
                 name='encoder',
                 reuse=False):
        self.input_shape = input_shape
        self.num_components = num_components
        self.num_outputs = num_outputs
        self.batch_size = batch_size
        self.regularizer = regularizer
        self.initializer = initializer

        self.dist_type = dist_type
        self.gate = gate

        if layer_sizes is None:
            layer_sizes = []

        with tf.variable_scope(name, reuse=reuse) as scope:
            super().__init__(scope)
            outputs = X

            for i, layer_size in enumerate(layer_sizes):
                with tf.variable_scope('fc_{}'.format(i)):
                    outputs = fully_connected_weight_norm(
                        inputs=outputs,
                        num_outputs=layer_size,
                        activation_fn=layer_activation_fn,
                        weights_regularizer=self.regularizer,
                        weights_initializer=self.initializer,
                        biases_initializer=tf.zeros_initializer())

            flatten_outputs = outputs
            self.build_prototypes(flatten_outputs, activation_fn)


class ConvPCD(DistBase):
    """ConvPCD

    Encode images to a latent space and
    produce `num_components` prototype vectors.
    """

    def __init__(self,
                 X,
                 input_shape,
                 num_components,
                 num_outputs,
                 batch_size,
                 gate=None,
                 dim=64,
                 max_dim=512,
                 min_dim=4,
                 dist_type='pcd',
                 initializer=tf.contrib.layers.xavier_initializer(),
                 regularizer=None,
                 layer_activation_fn=lrelu,
                 activation_fn=None,
                 name='encoder',
                 reuse=False):
        self.input_shape = input_shape
        self.num_components = num_components
        self.num_outputs = num_outputs
        self.batch_size = batch_size
        self.regularizer = regularizer
        self.initializer = initializer
        self.dist_type = dist_type
        self.gate = gate

        start_dim = min(input_shape[0], input_shape[1])
        nb_upconv = 0
        while start_dim % 2 == 0 and start_dim > min_dim:
            start_dim //= 2
            nb_upconv += 1

        with tf.variable_scope(name, reuse=reuse) as scope:
            super().__init__(scope)

            outputs = tf.reshape(X, (-1, ) + input_shape)

            # convolutions
            for i in range(nb_upconv):
                with tf.variable_scope('conv{}'.format(i + 1)):
                    outputs = conv2d_weight_norm(
                        inputs=outputs,
                        num_outputs=dim,
                        kernel_size=(5, 5),
                        stride=(2, 2),
                        padding='SAME',
                        activation_fn=lrelu,
                        weights_initializer=initializer,
                        weights_regularizer=regularizer)

                dim = min(dim * 2, max_dim)

            flatten_outputs = tf.contrib.layers.flatten(outputs)
            self.build_prototypes(flatten_outputs, activation_fn)
