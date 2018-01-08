"""Compatibility Family Learning for Item Recommendation and Generation
"""
import logging
import os
import gc

import tensorflow as tf

from tqdm import trange

from ..ops import arrange_grid
from ..ops import build_placeholder
from ..utils import dist_eval
from ..utils import load_best_stats
from ..utils import reduce_product
from .base import ModelBase
from .blocks import FCPCD
from .blocks import ConvTransposeGenerator
from .blocks import ConvPCD
from .blocks import Thresholder
from .blocks import ConvDiscriminator
from .blocks import FCDiscriminator
from .blocks import SRDiscriminator
from .blocks import SRGenerator

logger = logging.getLogger(__name__)


class SemiBase(ModelBase):
    def init(self, sess):
        pass

    def _set_data_params(self, is_double, disable_double, latent_shape,
                         source_shape, input_shape, ae_shape, batch_size,
                         data_norm, data_type, directed, data_directed,
                         train_data_transformer, val_data_transformer,
                         ae_transformer, data_normalizer, data_unnormalizer,
                         ae_normalizer, ae_unnormalizer, latent_normalizer):
        source_shape = source_shape if source_shape else input_shape
        self.is_double = is_double
        self.disable_double = disable_double
        self.latent_shape = latent_shape
        self.source_shape = source_shape
        self.source_size = reduce_product(source_shape)
        self.input_shape = input_shape
        self.input_size = reduce_product(input_shape)
        self.data_latent_size = reduce_product(
            latent_shape) if latent_shape else self.input_size
        self.ae_shape = ae_shape = ae_shape if ae_shape else input_shape
        self.ae_size = reduce_product(ae_shape)
        self.batch_size = batch_size
        self.data_norm = data_norm
        self.data_type = data_type
        self.directed = directed
        self.data_directed = data_directed

        self.train_data_transformer = train_data_transformer
        self.val_data_transformer = val_data_transformer
        self.ae_transformer = ae_transformer
        self.data_normalizer = data_normalizer
        self.data_unnormalizer = data_unnormalizer
        self.ae_normalizer = ae_normalizer
        self.ae_unnormalizer = ae_unnormalizer
        self.latent_normalizer = latent_normalizer

    def _build_regularizer(self):
        self.regularizer = tf.contrib.layers.l2_regularizer(
            self.reg_const) if self.reg_const > 0.0 else None

    def _build_z(self):
        z = tf.random_normal(
            (self.batch_size, self.z_dim),
            mean=0.0,
            stddev=self.z_stddev,
            name='z',
            dtype=tf.float32)
        self.zs = [
            tf.random_normal(
                (self.batch_size, self.z_dim),
                mean=0.0,
                stddev=self.z_stddev,
                name='z{}'.format(i),
                dtype=tf.float32) for i in range(self.num_components)
        ]

        self.z = tf.placeholder_with_default(
            z, (self.batch_size, self.z_dim), name='z')
        self.zs = [
            tf.placeholder_with_default(
                z, (self.batch_size, self.z_dim), name='z{}'.format(i))
            for i, z in enumerate(self.zs)
        ]
        self.z_hf = tf.slice(self.z, (0, 0), (self.batch_size // 2,
                                              self.z_dim))
        self.eps = tf.random_uniform(
            (self.batch_size, 1),
            minval=0.0,
            maxval=1.0,
            dtype=tf.float32,
            name='eps')

    def _build_train_data(self, batches):
        if self.is_double:
            self.input_data_pos_source = build_placeholder(
                [None, self.source_size], 'input_data_pos_source', batches[0])
            self.input_data_pos_target = build_placeholder(
                [None, self.source_size], 'input_data_pos_target', batches[2])
            self.input_data_neg_source = build_placeholder(
                [None, self.source_size], 'input_data_neg_source', batches[4])
            self.input_data_neg_target = build_placeholder(
                [None, self.source_size], 'input_data_neg_target', batches[6])

            if not self.disable_double:
                self.input_data_pos_source_latent = build_placeholder([
                    None, self.data_latent_size
                ], 'input_data_pos_source_latent', batches[1])
                self.input_data_pos_target_latent = build_placeholder([
                    None, self.data_latent_size
                ], 'input_data_pos_target_latent', batches[3])
                self.input_data_neg_source_latent = build_placeholder([
                    None, self.data_latent_size
                ], 'input_data_neg_source_latent', batches[5])
                self.input_data_neg_target_latent = build_placeholder([
                    None, self.data_latent_size
                ], 'input_data_neg_target_latent', batches[7])
        else:
            self.input_data_pos_source = build_placeholder(
                [None, self.source_size], 'input_data_pos_source', batches[0])
            self.input_data_pos_target = build_placeholder(
                [None, self.source_size], 'input_data_pos_target', batches[1])
            self.input_data_neg_source = build_placeholder(
                [None, self.source_size], 'input_data_neg_source', batches[2])
            self.input_data_neg_target = build_placeholder(
                [None, self.source_size], 'input_data_neg_target', batches[3])

        data_pos_source = self.train_data_transformer(
            self.input_data_pos_source
        ) if self.train_data_transformer else self.input_data_pos_source
        data_neg_source = self.train_data_transformer(
            self.input_data_neg_source
        ) if self.train_data_transformer else self.input_data_neg_source
        data_pos_target = self.train_data_transformer(
            self.input_data_pos_target
        ) if self.train_data_transformer else self.input_data_pos_target
        data_neg_target = self.train_data_transformer(
            self.input_data_neg_target
        ) if self.train_data_transformer else self.input_data_neg_target

        self.data_pos_source = self.data_normalizer(
            data_pos_source) if self.data_normalizer else data_pos_source
        self.data_neg_source = self.data_normalizer(
            data_neg_source) if self.data_normalizer else data_neg_source
        self.data_pos_target = self.data_normalizer(
            data_pos_target) if self.data_normalizer else data_pos_target
        self.data_neg_target = self.data_normalizer(
            data_neg_target) if self.data_normalizer else data_neg_target

        data_pos_ae_source = self.ae_transformer(
            data_pos_source) if self.ae_transformer else data_pos_source
        data_neg_ae_source = self.ae_transformer(
            data_neg_source) if self.ae_transformer else data_neg_source
        data_pos_ae_target = self.ae_transformer(
            data_pos_target) if self.ae_transformer else data_pos_target
        data_neg_ae_target = self.ae_transformer(
            data_neg_target) if self.ae_transformer else data_neg_target

        self.data_pos_ae_source = self.ae_normalizer(
            data_pos_ae_source) if self.ae_normalizer else data_pos_ae_source
        self.data_neg_ae_source = self.ae_normalizer(
            data_neg_ae_source) if self.ae_normalizer else data_neg_ae_source
        self.data_pos_ae_target = self.ae_normalizer(
            data_pos_ae_target) if self.ae_normalizer else data_pos_ae_target
        self.data_neg_ae_target = self.ae_normalizer(
            data_neg_ae_target) if self.ae_normalizer else data_neg_ae_target

        if self.is_double and not self.disable_double:
            self.data_pos_source_latent = self.latent_normalizer(
                self.input_data_pos_source_latent
            ) if self.latent_normalizer else self.input_data_pos_source_latent
            self.data_neg_source_latent = self.latent_normalizer(
                self.input_data_neg_source_latent
            ) if self.latent_normalizer else self.input_data_neg_source_latent
            self.data_pos_target_latent = self.latent_normalizer(
                self.input_data_pos_target_latent
            ) if self.latent_normalizer else self.input_data_pos_target_latent
            self.data_neg_target_latent = self.latent_normalizer(
                self.input_data_neg_target_latent
            ) if self.latent_normalizer else self.input_data_neg_target_latent
        else:
            self.data_pos_source_latent = self.data_pos_source
            self.data_neg_source_latent = self.data_neg_source
            self.data_pos_target_latent = self.data_pos_target
            self.data_neg_target_latent = self.data_neg_target

    def _build_unlabeled_data(self, batches):
        if self.is_double:
            self.input_data_unlabeled_source = build_placeholder(
                [None,
                 self.source_size], 'input_data_unlabeled_source', batches[0])

            self.input_data_unlabeled_target = build_placeholder(
                [None,
                 self.source_size], 'input_data_unlabeled_target', batches[3])

            if not self.disable_double:
                self.input_data_unlabeled_source_latent = build_placeholder([
                    None, self.data_latent_size
                ], 'input_data_unlabeled_source_latent', batches[1])

                self.input_data_unlabeled_target_latent = build_placeholder([
                    None, self.data_latent_size
                ], 'input_data_unlabeled_target_latent', batches[4])
        else:
            self.input_data_unlabeled_source = build_placeholder(
                [None,
                 self.source_size], 'input_data_unlabeled_source', batches[0])

            self.input_data_unlabeled_target = build_placeholder(
                [None,
                 self.source_size], 'input_data_unlabeled_target', batches[2])

        data_unlabeled_source = self.train_data_transformer(
            self.input_data_unlabeled_source
        ) if self.train_data_transformer else self.input_data_unlabeled_source

        data_unlabeled_ae_source = self.ae_transformer(
            data_unlabeled_source
        ) if self.ae_transformer else data_unlabeled_source

        self.data_unlabeled_ae_source = self.ae_normalizer(
            data_unlabeled_ae_source
        ) if self.ae_normalizer else data_unlabeled_ae_source

        self.data_unlabeled_source = self.data_normalizer(
            data_unlabeled_source
        ) if self.data_normalizer else data_unlabeled_source

        data_unlabeled_target = self.train_data_transformer(
            self.input_data_unlabeled_target
        ) if self.train_data_transformer else self.input_data_unlabeled_target
        data_unlabeled_target = self.train_data_transformer(
            self.input_data_unlabeled_target
        ) if self.train_data_transformer else self.input_data_unlabeled_target
        data_unlabeled_ae_target = self.ae_transformer(
            data_unlabeled_target
        ) if self.ae_transformer else data_unlabeled_target

        self.data_unlabeled_ae_target = self.ae_normalizer(
            data_unlabeled_ae_target
        ) if self.ae_normalizer else data_unlabeled_ae_target

        self.data_unlabeled_target = self.data_normalizer(
            data_unlabeled_target
        ) if self.data_normalizer else data_unlabeled_target

        if self.is_double and not self.disable_double:
            self.data_unlabeled_source_latent = self.latent_normalizer(
                self.input_data_unlabeled_source_latent
            ) if self.latent_normalizer else self.input_data_unlabeled_source_latent
            self.data_unlabeled_target_latent = self.latent_normalizer(
                self.input_data_unlabeled_target_latent
            ) if self.latent_normalizer else self.input_data_unlabeled_target_latent
        else:
            self.data_unlabeled_source_latent = self.data_unlabeled_source
            self.data_unlabeled_target_latent = self.data_unlabeled_target

    def _build_val_data(self, batches):
        if self.is_double:
            self.val_input_data_pos_source = build_placeholder(
                [None,
                 self.source_size], 'val_input_data_pos_source', batches[0])
            self.val_input_data_pos_target = build_placeholder(
                [None,
                 self.source_size], 'val_input_data_pos_target', batches[2])
            self.val_input_data_neg_source = build_placeholder(
                [None,
                 self.source_size], 'val_input_data_neg_source', batches[4])
            self.val_input_data_neg_target = build_placeholder(
                [None,
                 self.source_size], 'val_input_data_neg_target', batches[6])

            if not self.disable_double:
                self.val_input_data_pos_source_latent = build_placeholder([
                    None, self.data_latent_size
                ], 'val_input_data_pos_source_latent', batches[1])
                self.val_input_data_pos_target_latent = build_placeholder([
                    None, self.data_latent_size
                ], 'val_input_data_pos_target_latent', batches[3])
                self.val_input_data_neg_source_latent = build_placeholder([
                    None, self.data_latent_size
                ], 'val_input_data_neg_source_latent', batches[5])
                self.val_input_data_neg_target_latent = build_placeholder([
                    None, self.data_latent_size
                ], 'val_input_data_neg_target_latent', batches[7])
        else:
            self.val_input_data_pos_source = build_placeholder(
                [None,
                 self.source_size], 'val_input_data_pos_source', batches[0])
            self.val_input_data_pos_target = build_placeholder(
                [None,
                 self.source_size], 'val_input_data_pos_target', batches[1])
            self.val_input_data_neg_source = build_placeholder(
                [None,
                 self.source_size], 'val_input_data_neg_source', batches[2])
            self.val_input_data_neg_target = build_placeholder(
                [None,
                 self.source_size], 'val_input_data_neg_target', batches[3])

        val_data_pos_source = self.val_data_transformer(
            self.val_input_data_pos_source
        ) if self.val_data_transformer else self.val_input_data_pos_source
        val_data_neg_source = self.val_data_transformer(
            self.val_input_data_neg_source
        ) if self.val_data_transformer else self.val_input_data_neg_source
        val_data_pos_target = self.val_data_transformer(
            self.val_input_data_pos_target
        ) if self.val_data_transformer else self.val_input_data_pos_target
        val_data_neg_target = self.val_data_transformer(
            self.val_input_data_neg_target
        ) if self.val_data_transformer else self.val_input_data_neg_target

        self.val_data_pos_source = self.data_normalizer(
            val_data_pos_source
        ) if self.data_normalizer else val_data_pos_source
        self.val_data_neg_source = self.data_normalizer(
            val_data_neg_source
        ) if self.data_normalizer else val_data_neg_source
        self.val_data_pos_target = self.data_normalizer(
            val_data_pos_target
        ) if self.data_normalizer else val_data_pos_target
        self.val_data_neg_target = self.data_normalizer(
            val_data_neg_target
        ) if self.data_normalizer else val_data_neg_target

        self.val_data_pos_ae_source = self.ae_normalizer(
            val_data_pos_source) if self.ae_normalizer else val_data_pos_source
        self.val_data_neg_ae_source = self.ae_normalizer(
            val_data_neg_source) if self.ae_normalizer else val_data_neg_source
        self.val_data_pos_ae_target = self.ae_normalizer(
            val_data_pos_target) if self.ae_normalizer else val_data_pos_target
        self.val_data_neg_ae_target = self.ae_normalizer(
            val_data_neg_target) if self.ae_normalizer else val_data_neg_target

        if self.is_double and not self.disable_double:
            self.val_data_pos_source_latent = self.latent_normalizer(
                self.val_input_data_pos_source_latent
            ) if self.latent_normalizer else self.val_input_data_pos_source_latent
            self.val_data_neg_source_latent = self.latent_normalizer(
                self.val_input_data_neg_source_latent
            ) if self.latent_normalizer else self.val_input_data_neg_source_latent
            self.val_data_pos_target_latent = self.latent_normalizer(
                self.val_input_data_pos_target_latent
            ) if self.latent_normalizer else self.val_input_data_pos_target_latent
            self.val_data_neg_target_latent = self.latent_normalizer(
                self.val_input_data_neg_target_latent
            ) if self.latent_normalizer else self.val_input_data_neg_target_latent
        else:
            self.val_data_pos_source_latent = self.val_data_pos_source
            self.val_data_neg_source_latent = self.val_data_neg_source
            self.val_data_pos_target_latent = self.val_data_pos_target
            self.val_data_neg_target_latent = self.val_data_neg_target


class CFL(SemiBase):
    def init(self, sess):
        pass

    def get_name(self, no_gan=False):
        name = 'cfl'
        name += '_' + self.dist_type
        name += '_' + self.model_type
        if self.directed:
            name += '_di'
        if self.pos_weight:
            name += '_pw_{}'.format(self.pos_weight)
        if self.caffe_margin:
            name += '_margin_{}'.format(self.caffe_margin)
        name += '_' + self.data_type
        name += '_ls_{}'.format(self.latent_size)
        if self.dist_type != 'siamese':
            name += '_nc_{}'.format(self.num_components)
        if self.act_type:
            name += '_act_{}'.format(self.act_type)
        if self.disable_double:
            name += '_dd'
        if self.use_threshold:
            name += '_ut'
        if self.reg_const:
            name += '_reg_{}'.format(self.reg_const)
        if self.data_norm:
            name += '_norm_{}'.format(
                '_'.join(str(norm) for norm in self.data_norm))
        if self.lambda_m:
            name += '_lm_{}'.format(self.lambda_m)
        if self.gan and not no_gan:
            if self.cgan:
                name += '_cgan_z_{}'.format(self.z_dim)
                if self.t_dim:
                    name += '_t_{}'.format(self.t_dim)
            else:
                name += '_gan_z_{}'.format(self.z_dim)
                if self.m_prj:
                    name += '_m_prj_{}'.format(self.m_prj)
                if self.m_enc:
                    name += '_m_enc_{}'.format(self.m_enc)
            if self.lambda_gp:
                name += '_dra_{}_{}'.format(self.lambda_gp, self.lambda_dra)
            if self.gan_type != 'conv':
                name += '_' + self.gan_type
        if self.run_tag:
            name += '_run_' + self.run_tag
        return name

    def __init__(self,
                 is_double,
                 disable_double,
                 latent_shape,
                 source_shape,
                 input_shape,
                 ae_shape,
                 batch_size,
                 data_norm,
                 data_type,
                 num_components,
                 pos_weight,
                 latent_size,
                 caffe_margin,
                 gan,
                 cgan,
                 t_dim,
                 dist_type,
                 act_type,
                 use_threshold,
                 lr,
                 beta1,
                 beta2,
                 z_dim,
                 z_stddev,
                 g_dim,
                 g_lr,
                 g_beta1,
                 g_beta2,
                 m_prj,
                 m_enc,
                 d_dim,
                 d_lr,
                 d_beta1,
                 d_beta2,
                 lambda_gp,
                 lambda_m,
                 lambda_dra,
                 directed,
                 data_directed,
                 model_type,
                 gan_type,
                 reg_const,
                 batches=None,
                 val_batches=None,
                 unlabeled_batches=None,
                 train_data_transformer=None,
                 val_data_transformer=None,
                 ae_transformer=None,
                 data_normalizer=None,
                 data_unnormalizer=None,
                 ae_normalizer=None,
                 ae_unnormalizer=None,
                 latent_normalizer=None,
                 run_tag=None,
                 name='CFL',
                 reuse=False):
        self._set_data_params(
            is_double=is_double,
            disable_double=disable_double,
            latent_shape=latent_shape,
            source_shape=source_shape,
            input_shape=input_shape,
            ae_shape=ae_shape,
            batch_size=batch_size,
            data_norm=data_norm,
            data_type=data_type,
            directed=directed,
            data_directed=data_directed,
            train_data_transformer=train_data_transformer,
            val_data_transformer=val_data_transformer,
            ae_transformer=ae_transformer,
            data_normalizer=data_normalizer,
            data_unnormalizer=data_unnormalizer,
            ae_normalizer=ae_normalizer,
            ae_unnormalizer=ae_unnormalizer,
            latent_normalizer=latent_normalizer)

        self.pos_weight = pos_weight
        self.num_components = num_components
        self.latent_size = latent_size
        self.caffe_margin = caffe_margin
        self.gan = gan
        self.cgan = cgan
        self.t_dim = t_dim
        self.dist_type = dist_type
        self.act_type = act_type
        self.use_threshold = use_threshold
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.z_dim = z_dim
        self.z_stddev = z_stddev
        self.g_dim = g_dim
        self.g_lr = g_lr
        self.g_beta1 = g_beta1
        self.g_beta2 = g_beta2
        self.m_prj = m_prj
        self.m_enc = m_enc
        self.d_dim = d_dim
        self.d_lr = d_lr
        self.d_beta1 = d_beta1
        self.d_beta2 = d_beta2
        self.lambda_gp = lambda_gp
        self.lambda_m = lambda_m
        self.lambda_dra = lambda_dra
        self.model_type = model_type
        self.gan_type = gan_type
        self.reg_const = reg_const
        self.run_tag = run_tag

        with tf.variable_scope(name, reuse=reuse) as scope:
            super().__init__(scope)

            self.ema = tf.train.ExponentialMovingAverage(decay=0.99)
            self.ema_ops = []

            # random source
            if self.gan:
                self._build_z()

            with tf.variable_scope('gate'):
                c = tf.random_uniform(
                    (batch_size, ),
                    minval=0,
                    maxval=num_components,
                    dtype=tf.int32,
                    name='c')
                self.c = tf.placeholder_with_default(
                    c, (self.batch_size, ), name='c')

                batch_range = tf.range(batch_size)
                self.gate = tf.stack([batch_range, self.c], axis=1)

            self._build_regularizer()

            # train data input
            if batches is None:
                batches = [None, None, None, None]
                if self.is_double:
                    batches *= 2
            self._build_train_data(batches)

            if unlabeled_batches is None:
                unlabeled_batches = [None, None, None, None]
                if self.is_double:
                    unlabeled_batches.extend([None, None])
            self._build_unlabeled_data(unlabeled_batches)

            # val data input
            if val_batches is None:
                val_batches = [None, None, None, None]
                if self.is_double:
                    val_batches *= 2
            self._build_val_data(val_batches)

            self._build_model(reuse=reuse)
            self._build_losses()
            self._build_stats()
            self._build_optimizer()
            self._build_summary()

    def dist_fn(self, inputs, name, reuse=False):
        gate = self.gate if self.gan else None
        input_shape = self.input_shape if not self.is_double else self.latent_shape
        activation_fns = {
            'linear': None,
            'tanh': tf.nn.tanh,
            'sigmoid': tf.nn.sigmoid,
            'relu': tf.nn.relu
        }
        activation_fn = activation_fns[
            self.act_type] if self.act_type else None
        if self.model_type == 'linear':
            return FCPCD(
                inputs,
                input_shape=input_shape,
                num_components=self.num_components,
                num_outputs=self.latent_size,
                batch_size=self.batch_size,
                gate=gate,
                dist_type=self.dist_type,
                regularizer=self.regularizer,
                activation_fn=activation_fn,
                name=name,
                reuse=reuse)
        elif self.model_type == 'conv':
            return ConvPCD(
                inputs,
                input_shape=input_shape,
                num_components=self.num_components,
                num_outputs=self.latent_size,
                batch_size=self.batch_size,
                gate=gate,
                dist_type=self.dist_type,
                regularizer=self.regularizer,
                activation_fn=activation_fn,
                name=name,
                reuse=reuse)

    def disc_fn(self,
                inputs,
                t=None,
                t_dim=None,
                name='Discriminator',
                reuse=False):
        disc_clss = {'conv': ConvDiscriminator, 'srgan': SRDiscriminator}
        disc_cls = disc_clss[self.gan_type]
        return disc_cls(
            inputs,
            num_classes=None,
            latent_size=self.latent_size,
            input_shape=self.ae_shape,
            batch_size=self.batch_size,
            regularizer=None,
            t=t,
            t_dim=t_dim,
            disc_activation_fn=tf.nn.sigmoid,
            latent_activation_fn=None,
            name=name,
            reuse=reuse)

    def decode_fn(self, z, name='DistDecoder', reuse=False):
        activation_fns = {
            'linear': None,
            'tanh': tf.nn.tanh,
            'sigmoid': tf.nn.sigmoid,
            'relu': tf.nn.relu
        }
        activation_fn = activation_fns[self.data_type]
        return ConvTransposeGenerator(
            zs=z,
            input_shape=self.ae_shape,
            batch_size=self.batch_size,
            regularizer=None,
            activation_fn=activation_fn,
            name=name,
            reuse=reuse)

    def gen_fn(self, z, c, t_dim=None, name='Generator', reuse=False):
        activation_fns = {
            'linear': None,
            'tanh': tf.nn.tanh,
            'sigmoid': tf.nn.sigmoid,
            'relu': tf.nn.relu
        }
        activation_fn = activation_fns[self.data_type]
        gen_clss = {'conv': ConvTransposeGenerator, 'srgan': SRGenerator}
        gen_cls = gen_clss[self.gan_type]
        return gen_cls(
            zs=[z, c],
            t_dim=t_dim,
            input_shape=self.ae_shape,
            batch_size=self.batch_size,
            regularizer=None,
            activation_fn=activation_fn,
            name=name,
            reuse=reuse)

    def thres_fn(self, inputs, reuse=False):
        return Thresholder(inputs, reuse=reuse)

    def _build_names(self):
        if self.directed:
            self.dist_name_src = 'DistEncoderSrc'
            self.dist_name_dst = 'DistEncoderDst'
        else:
            self.dist_name_src = self.dist_name_dst = 'DistEncoder'

    def _build_model(self, reuse):
        self._build_names()

        # -- train --
        self.s_pos_src = self.dist_fn(
            self.data_pos_source_latent, name=self.dist_name_src)
        self.s_pos_target = self.dist_fn(
            self.data_pos_target_latent,
            name=self.dist_name_dst,
            reuse=not self.directed)
        self.s_pos_dists = self.s_pos_src.build_dist(self.s_pos_target)
        self.s_pos_predicts = self.thres_fn(self.s_pos_dists)

        self.s_neg_src = self.dist_fn(
            self.data_neg_source_latent, name=self.dist_name_src, reuse=True)
        self.s_neg_target = self.dist_fn(
            self.data_neg_target_latent, name=self.dist_name_dst, reuse=True)
        self.s_neg_dists = self.s_neg_src.build_dist(self.s_neg_target)
        self.s_neg_predicts = self.thres_fn(self.s_neg_dists, reuse=True)

        # -- val --
        self.val_s_pos_src = self.dist_fn(
            self.val_data_pos_source_latent,
            name=self.dist_name_src,
            reuse=True)
        self.val_s_pos_target = self.dist_fn(
            self.val_data_pos_target_latent,
            name=self.dist_name_dst,
            reuse=True)
        self.val_s_pos_dists = self.val_s_pos_src.build_dist(
            self.val_s_pos_target)
        self.val_s_pos_predicts = self.thres_fn(
            self.val_s_pos_dists, reuse=True)

        self.val_s_neg_src = self.dist_fn(
            self.val_data_neg_source_latent,
            name=self.dist_name_src,
            reuse=True)
        self.val_s_neg_target = self.dist_fn(
            self.val_data_neg_target_latent,
            name=self.dist_name_dst,
            reuse=True)
        self.val_s_neg_dists = self.val_s_neg_src.build_dist(
            self.val_s_neg_target)
        self.val_s_neg_predicts = self.thres_fn(
            self.val_s_neg_dists, reuse=True)

        if self.gan:
            self.s_encoder = self.dist_fn(
                self.data_unlabeled_target_latent,
                name=self.dist_name_dst,
                reuse=True)
            self.s_g = self.dist_fn(
                self.data_unlabeled_source_latent,
                name=self.dist_name_src,
                reuse=True)

        if self.gan:

            def _perturb(X):
                mean, var = tf.nn.moments(X, list(range(len(X.shape))))
                std = tf.sqrt(var)
                return X + self.lambda_dra * std * self.eps

            if self.cgan:
                pos_src = self.data_pos_source_latent if self.t_dim else self.s_pos_src.activations
                neg_src = self.data_neg_source_latent if self.t_dim else self.s_neg_src.activations
                self.g = self.gen_fn(z=self.z, c=pos_src, t_dim=self.t_dim)

                assert self.batch_size % 2 == 0
                half_size = self.batch_size // 2

                self.d_real = self.disc_fn(
                    self.data_pos_ae_target, t=pos_src, t_dim=self.t_dim)
                self.d_fake = self.disc_fn(
                    self.g.activations,
                    t=pos_src,
                    t_dim=self.t_dim,
                    reuse=True)
                self.d_neg = self.disc_fn(
                    self.data_neg_ae_target,
                    t=neg_src,
                    t_dim=self.t_dim,
                    reuse=True)

                half1, half2 = tf.split(
                    pos_src, [half_size, half_size], axis=0)
                half_latent = (half1 + half2) / 2.
                self.g_int = self.gen_fn(
                    z=self.z_hf, c=half_latent, reuse=True, t_dim=self.t_dim)
                self.d_int = self.disc_fn(
                    self.g_int.activations,
                    t=half_latent,
                    reuse=True,
                    t_dim=self.t_dim)

                if self.lambda_gp:
                    self.X_hat = _perturb(self.data_pos_ae_target)
                    self.d_hat = self.disc_fn(
                        self.X_hat, t=pos_src, t_dim=self.t_dim, reuse=True)
            else:
                self.g = self.gen_fn(
                    z=self.z, c=self.s_encoder.activations, t_dim=self.t_dim)
                self.g_neg = self.gen_fn(
                    z=self.z,
                    c=self.s_neg_src.one_prototype_activations,
                    t_dim=self.t_dim,
                    reuse=True)

                self.d_real = self.disc_fn(self.data_unlabeled_ae_target)
                self.d_fake = self.disc_fn(self.g.activations, reuse=True)

                self.g_prj = self.gen_fn(
                    z=self.z,
                    c=self.s_g.one_prototype_activations,
                    t_dim=self.t_dim,
                    reuse=True)
                self.d_prj = self.disc_fn(self.g_prj.activations, reuse=True)

                self.d_neg = self.disc_fn(self.g_neg.activations, reuse=True)

                if self.lambda_gp:
                    self.X_hat = _perturb(self.data_unlabeled_ae_target)
                    self.d_hat = self.disc_fn(self.X_hat, reuse=True)

            def _dup_dims(inputs, num=10):
                assert self.batch_size % num == 0
                outputs = tf.slice(inputs, (0, 0), (num,
                                                    self.data_latent_size))
                return tf.tile(outputs, [self.batch_size // num, 1])

            # sample encoder for display prototypes
            self.s_encoder_sample_latent = _dup_dims(
                self.val_data_pos_source_latent)
            self.s_encoder_sample = self.dist_fn(
                self.s_encoder_sample_latent,
                name=self.dist_name_src,
                reuse=True)
            self.s_encoder_target_sample = self.dist_fn(
                _dup_dims(self.val_data_pos_target_latent),
                name=self.dist_name_dst,
                reuse=True)

            self.g_target = self.gen_fn(
                z=self.z,
                c=self.s_encoder_target_sample.activations,
                reuse=True)

            if self.cgan:
                src_latent = self.s_encoder_sample_latent if self.t_dim else self.s_encoder_sample.activations
                self.g_prototypes = [
                    self.gen_fn(
                        z=self.zs[i],
                        c=src_latent,
                        t_dim=self.t_dim,
                        reuse=True) for i in range(self.num_components)
                ]
                self.d_prototypes = [
                    self.disc_fn(
                        g_prototype.activations,
                        t=src_latent,
                        t_dim=self.t_dim,
                        reuse=True) for g_prototype in self.g_prototypes
                ]
            else:
                self.g_prototypes = [
                    self.gen_fn(
                        z=self.z,
                        c=one_prototype_activations,
                        t_dim=self.t_dim,
                        reuse=True)
                    for one_prototype_activations in
                    self.s_encoder_sample.all_prototype_activations
                ]
                self.d_prototypes = [
                    self.disc_fn(g_prototype.activations, reuse=True)
                    for g_prototype in self.g_prototypes
                ]

    def _build_losses(self):
        self._build_dist_losses()

        if self.gan:
            self._build_gan_losses()

    def _build_dist_losses(self):
        # regularizations
        if self.directed:
            self.s_loss_reg = self.s_pos_src.reg_loss(
            ) + self.s_pos_target.reg_loss()
        else:
            self.s_loss_reg = self.s_pos_src.reg_loss()
        self.s_total_loss = self.s_loss_reg
        self.pre_s_total_loss = tf.constant(0.)

        # sigmoid thres loss
        self.s_p_loss_pos = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                logits=self.s_pos_predicts.outputs,
                labels=tf.ones_like(self.s_pos_predicts.outputs)))
        self.s_p_loss_neg = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                logits=self.s_neg_predicts.outputs,
                labels=tf.zeros_like(self.s_neg_predicts.outputs)))
        if self.pos_weight:
            self.s_thres_loss = (
                self.s_p_loss_pos * self.pos_weight + self.s_p_loss_neg)
        else:
            self.s_thres_loss = self.s_p_loss_pos + self.s_p_loss_neg
        if self.use_threshold:
            self.s_total_loss += self.s_thres_loss

        # contrastive loss
        self.s_margins = tf.reduce_mean(self.s_pos_dists - self.s_neg_dists)
        self.s_pos_dists_adapt = tf.reduce_mean(
            tf.sqrt(self.s_pos_dists + 1e-7))
        self.s_neg_dists_adapt = tf.reduce_mean(
            tf.sqrt(self.s_neg_dists + 1e-7))
        self.s_margin_adapt = (
            self.s_pos_dists_adapt + self.s_neg_dists_adapt) / 2.0
        self.ema_ops.append(
            self.ema.apply([
                self.s_pos_dists_adapt, self.s_neg_dists_adapt,
                self.s_margin_adapt
            ]))
        self.s_margin_adapt_avg = self.ema.average(self.s_margin_adapt)
        self.s_pos_dists_adapt_avg = self.ema.average(self.s_pos_dists_adapt)
        self.s_neg_dists_adapt_avg = self.ema.average(self.s_neg_dists_adapt)

        if self.caffe_margin:
            # caffe margin mode
            # ref: https://github.com/BVLC/caffe/issues/2308#issuecomment-92660209
            self.s_cd_loss_pos = tf.reduce_mean(self.s_pos_dists)
            if self.pos_weight:
                self.s_cd_loss_pos = self.s_cd_loss_pos * self.pos_weight
            self.s_cd_loss_neg = tf.reduce_mean(
                tf.maximum(0., self.caffe_margin - self.s_neg_dists))
            self.s_cd_loss = 0.5 * (self.s_cd_loss_pos + self.s_cd_loss_neg)
            self.s_total_loss += self.s_cd_loss
        elif self.lambda_m:
            self.s_cd_loss_neg = tf.constant(0.0)
            self.s_cd_loss_pos = tf.reduce_mean(
                self.s_pos_dists) * self.lambda_m
            if self.pos_weight:
                self.s_cd_loss_pos = self.s_cd_loss_pos * self.pos_weight
            self.s_cd_loss = self.s_cd_loss_pos
            self.s_total_loss += self.s_cd_loss

        # prediction accuracy
        correct_prediction_pos = tf.greater(self.s_pos_predicts.outputs, 0.0)
        correct_prediction_neg = tf.less_equal(self.s_neg_predicts.outputs,
                                               0.0)
        self.s_accuracy = (
            tf.reduce_mean(tf.cast(correct_prediction_pos, tf.float32)) +
            tf.reduce_mean(tf.cast(correct_prediction_neg, tf.float32))) / 2.0
        self.ema_ops.append(self.ema.apply([self.s_accuracy]))
        self.s_accuracy_avg = self.ema.average(self.s_accuracy)

        correct_prediction_pos = tf.greater(self.val_s_pos_predicts.outputs,
                                            0.0)
        correct_prediction_neg = tf.less_equal(self.val_s_neg_predicts.outputs,
                                               0.0)
        self.val_s_accuracy = (
            tf.reduce_mean(tf.cast(correct_prediction_pos, tf.float32)) +
            tf.reduce_mean(tf.cast(correct_prediction_neg, tf.float32))) / 2.0
        self.ema_ops.append(self.ema.apply([self.val_s_accuracy]))
        self.val_s_accuracy_avg = self.ema.average(self.val_s_accuracy)

    def _build_gan_losses(self):
        with tf.name_scope('disc_loss'):
            self.d_loss_reg = self.d_real.reg_loss()
            self.d_total_loss = self.d_loss_reg
            self.d_loss_real = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(
                    logits=self.d_real.disc_outputs,
                    labels=tf.ones_like(self.d_real.disc_outputs)))
            if not self.cgan:
                self.d_loss_enc = tf.reduce_mean(
                    tf.nn.sigmoid_cross_entropy_with_logits(
                        logits=self.d_fake.disc_outputs,
                        labels=tf.zeros_like(self.d_fake.disc_outputs)))
                self.d_loss_prj = tf.reduce_mean(
                    tf.nn.sigmoid_cross_entropy_with_logits(
                        logits=self.d_prj.disc_outputs,
                        labels=tf.zeros_like(self.d_prj.disc_outputs)))
                self.d_loss_fake = 0.5 * (self.d_loss_enc + self.d_loss_prj)
            else:
                self.d_loss_fake = tf.reduce_mean(
                    tf.nn.sigmoid_cross_entropy_with_logits(
                        logits=self.d_fake.disc_outputs,
                        labels=tf.zeros_like(self.d_fake.disc_outputs)))

            if self.cgan:
                self.d_loss_neg = tf.reduce_mean(
                    tf.nn.sigmoid_cross_entropy_with_logits(
                        logits=self.d_neg.disc_outputs,
                        labels=tf.zeros_like(self.d_neg.disc_outputs)))
                self.d_loss = self.d_loss_real + (self.d_loss_fake +
                                                  self.d_loss_neg) / 2.
            else:
                self.d_loss = self.d_loss_real + self.d_loss_fake
            self.d_total_loss += self.d_loss

            if self.lambda_gp:
                d_grad = tf.gradients(self.d_hat.disc_outputs, [self.X_hat])[0]
                self.d_grad_loss = self.lambda_gp * tf.reduce_mean(
                    tf.square(
                        tf.sqrt(tf.reduce_sum(tf.square(d_grad), 1)) - 1.0))
                self.d_total_loss += self.d_grad_loss

            self.d_real_accuracy = tf.reduce_mean(
                tf.cast(tf.greater(self.d_real.disc_outputs, 0.0), tf.float32))
            self.d_fake_accuracy = tf.reduce_mean(
                tf.cast(
                    tf.less_equal(self.d_fake.disc_outputs, 0.0), tf.float32))

            # aux loss
            if not self.cgan:
                self.d_loss_d = tf.reduce_mean(
                    tf.reduce_sum(
                        tf.square(self.d_real.latent_activations -
                                  self.s_encoder.activations),
                        axis=-1))
                self.d_total_loss += self.d_loss_d

        with tf.name_scope('gen_loss'):
            self.g_loss_reg = self.g.reg_loss()
            self.g_total_loss = self.g_loss_reg

            if not self.cgan:
                self.g_loss_enc = tf.reduce_mean(
                    tf.nn.sigmoid_cross_entropy_with_logits(
                        logits=self.d_fake.disc_outputs,
                        labels=tf.ones_like(self.d_fake.disc_outputs)))
                self.g_loss_prj = tf.reduce_mean(
                    tf.nn.sigmoid_cross_entropy_with_logits(
                        logits=self.d_prj.disc_outputs,
                        labels=tf.ones_like(self.d_prj.disc_outputs)))
                self.g_loss = 0.5 * (self.g_loss_prj + self.g_loss_enc)
            else:
                self.g_loss = tf.reduce_mean(
                    tf.nn.sigmoid_cross_entropy_with_logits(
                        logits=self.d_fake.disc_outputs,
                        labels=tf.ones_like(self.d_fake.disc_outputs)))
            self.g_total_loss += self.g_loss

            self.g_accuracy = tf.reduce_mean(
                tf.cast(tf.greater(self.d_fake.disc_outputs, 0.0), tf.float32))

            # aux loss
            if self.cgan:
                self.g_loss_int = tf.reduce_mean(
                    tf.nn.sigmoid_cross_entropy_with_logits(
                        logits=self.d_int.disc_outputs,
                        labels=tf.ones_like(self.d_int.disc_outputs)))
                self.g_total_loss += self.g_loss_int
            else:
                self.g_d_enc = tf.reduce_sum(
                    tf.square(self.d_fake.latent_activations -
                              self.s_encoder.activations),
                    axis=-1)
                self.g_d_neg = tf.reduce_sum(
                    tf.square(self.d_neg.latent_activations -
                              self.s_neg_target.activations),
                    axis=-1)

                if self.m_enc:
                    self.g_loss_d = tf.reduce_mean(
                        tf.square(
                            tf.maximum(0., -self.m_enc + tf.sqrt(self.g_d_enc +
                                                                 1e-7))))
                else:
                    self.g_loss_d = tf.reduce_mean(self.g_d_enc)
                self.g_total_loss += self.g_loss_d

                if self.m_prj:
                    self.g_loss_d_neg = tf.reduce_mean(
                        tf.square(
                            tf.maximum(0., self.m_prj - tf.sqrt(self.g_d_neg +
                                                                1e-7))))
                    self.g_total_loss += self.g_loss_d_neg

    def _build_main_optimizer(self):
        if self.directed:
            self.s_vars = self.s_pos_src.get_vars(
            ) + self.s_pos_target.get_vars()
        else:
            self.s_vars = self.s_neg_src.get_vars()
        self.th_vars = self.s_pos_predicts.get_vars()

        s_total_vars = []
        s_total_vars.extend(self.s_vars)

        if not self.use_threshold:
            self.th_optim = tf.train.AdamOptimizer(
                self.lr, beta1=self.beta1, beta2=self.beta2).minimize(
                    self.s_thres_loss, var_list=self.th_vars)
        else:
            s_total_vars.extend(self.th_vars)

        self.s_optim = tf.train.AdamOptimizer(
            self.lr, beta1=self.beta1, beta2=self.beta2).minimize(
                self.s_total_loss, var_list=s_total_vars)

    def _build_post_optimizer(self):
        self.g_vars = self.g.get_vars()
        self.d_vars = self.d_real.get_vars()
        self.post_g_optim = tf.train.AdamOptimizer(
            self.g_lr, beta1=self.g_beta1, beta2=self.g_beta2).minimize(
                self.g_total_loss, var_list=self.g_vars)

        self.post_d_optim = tf.train.AdamOptimizer(
            self.d_lr, beta1=self.d_beta1, beta2=self.d_beta2).minimize(
                self.d_total_loss, var_list=self.d_vars)

    def _build_stats(self):
        self.update_stats = tf.group(*self.ema_ops)

    def _build_optimizer(self):
        self._build_main_optimizer()

        if self.gan:
            self._build_post_optimizer()

    def _build_summary(self):
        self.image_summary = []
        self.summary = []
        self.post_image_summary = []
        self.post_summary = []
        with tf.name_scope('summary'):
            with tf.name_scope('dist'):
                self._build_dist_summary([self.summary])
            if self.gan:
                with tf.name_scope('gan'):
                    self._build_gan_summary([self.post_summary])

        self.summary = tf.summary.merge(self.summary, name='summary')
        if self.gan:
            self.post_summary = tf.summary.merge(
                self.post_summary, name='post_summary')

        if len(self.input_shape) > 1:
            with tf.name_scope('image_summary'):
                self._build_image_summary(
                    [self.image_summary, self.post_image_summary])
                if self.gan:
                    with tf.name_scope('gan'):
                        self._build_gan_image_summary(
                            [self.post_image_summary])

            self.all_summary = tf.summary.merge(
                [self.summary] + self.image_summary, name='all_summary')
            if self.gan:
                self.post_all_summary = tf.summary.merge(
                    [self.post_summary] + self.post_image_summary,
                    name='post_all_summary')

        else:
            self.all_summary = self.summary
            if self.gan:
                self.post_all_summary = self.post_summary

    def _build_dist_summary(self, summary_list):
        self.add_summary(
            's_pos_dists',
            tf.reduce_mean(self.s_pos_dists),
            summary_list=summary_list)
        self.add_summary(
            's_neg_dists',
            tf.reduce_mean(self.s_neg_dists),
            summary_list=summary_list)
        self.add_summary(
            'threshold',
            self.s_pos_predicts.threshold,
            summary_list=summary_list)
        self.add_summary(
            's_pos_predicts',
            tf.reduce_mean(self.s_pos_predicts.outputs),
            summary_list=summary_list)
        self.add_summary(
            's_neg_predicts',
            tf.reduce_mean(self.s_neg_predicts.outputs),
            summary_list=summary_list)
        self.add_summary(
            's_margins', self.s_margins, summary_list=summary_list)

        self.add_summary(
            's_thres_loss', self.s_thres_loss, summary_list=summary_list)
        self.add_summary(
            's_total_loss', self.s_total_loss, summary_list=summary_list)

        self.add_summary(
            's_accuracy', self.s_accuracy, summary_list=summary_list)
        self.add_summary(
            'val_s_accuracy', self.val_s_accuracy, summary_list=summary_list)

        self.add_summary(
            's_margin_adapt_avg',
            self.s_margin_adapt_avg,
            summary_list=summary_list)
        if self.caffe_margin or self.lambda_m:
            self.add_summary(
                's_cd_loss', self.s_cd_loss, summary_list=summary_list)
            self.add_summary(
                's_cd_loss_neg', self.s_cd_loss_neg, summary_list=summary_list)
            self.add_summary(
                's_cd_loss_pos', self.s_cd_loss_pos, summary_list=summary_list)

    def _build_gan_summary(self, summary_list):
        self.add_summary('d_loss', self.d_loss, summary_list=summary_list)
        self.add_summary(
            'd_real_accuracy', self.d_real_accuracy, summary_list=summary_list)
        self.add_summary(
            'd_fake_accuracy', self.d_fake_accuracy, summary_list=summary_list)
        if not self.cgan:
            self.add_summary(
                'd_loss_prj', self.d_loss_prj, summary_list=summary_list)
            self.add_summary(
                'd_loss_enc', self.d_loss_enc, summary_list=summary_list)
            self.add_summary(
                'd_loss_d', self.d_loss_d, summary_list=summary_list)

        self.add_summary('g_loss', self.g_loss, summary_list=summary_list)
        if not self.cgan:
            self.add_summary(
                'g_loss_prj', self.g_loss_prj, summary_list=summary_list)
            self.add_summary(
                'g_loss_enc', self.g_loss_enc, summary_list=summary_list)
            self.add_summary(
                'g_loss_d', self.g_loss_d, summary_list=summary_list)
        self.add_summary(
            'g_accuracy', self.g_accuracy, summary_list=summary_list)

        if self.lambda_gp:
            self.add_summary(
                'd_grad_loss', self.d_grad_loss, summary_list=summary_list)

        self.add_summary(
            's_latent',
            self.s_encoder.activations,
            tf.summary.histogram,
            summary_list=summary_list)
        if self.is_double:
            self.add_summary(
                'data_pos_source_latent',
                self.data_pos_source_latent,
                tf.summary.histogram,
                summary_list=summary_list)

        if self.m_prj and not self.cgan:
            self.add_summary(
                'g_loss_d_neg', self.g_loss_d_neg, summary_list=summary_list)
            self.add_summary(
                'g_loss_cd',
                self.g_loss_d + self.g_loss_d_neg,
                summary_list=summary_list)

    def _build_image_summary(self, summary_list):
        num_cols = 10
        num_rows = 8
        rows = []
        rows.append(tf.slice(self.data_pos_source, (0, 0), (num_cols, -1)))
        rows.append(tf.slice(self.data_pos_target, (0, 0), (num_cols, -1)))
        rows.append(tf.slice(self.data_neg_source, (0, 0), (num_cols, -1)))
        rows.append(tf.slice(self.data_neg_target, (0, 0), (num_cols, -1)))
        rows.append(tf.slice(self.val_data_pos_source, (0, 0), (num_cols, -1)))
        rows.append(tf.slice(self.val_data_pos_target, (0, 0), (num_cols, -1)))
        rows.append(tf.slice(self.val_data_neg_source, (0, 0), (num_cols, -1)))
        rows.append(tf.slice(self.val_data_neg_target, (0, 0), (num_cols, -1)))
        rows = tf.concat(rows, 0)
        rows = self.data_unnormalizer(rows)
        rows = arrange_grid(rows, num_rows, num_cols, self.input_shape)
        self.add_summary(
            'train_val_grid',
            rows,
            tf.summary.image,
            summary_list=summary_list)

    def _build_gan_image_summary(self, summary_list):
        num_cols = 10
        num_rows_each = (self.batch_size // num_cols)
        num_rows = self.num_components * num_rows_each + 1
        self.add_summary(
            'image_hist',
            self.val_data_pos_ae_source,
            tf.summary.histogram,
            summary_list=summary_list)
        self.add_summary(
            'gan_hist',
            self.g_prototypes[0].activations,
            tf.summary.histogram,
            summary_list=summary_list)
        rows = []
        rows.append(
            tf.slice(self.val_data_pos_ae_source, (0, 0), (num_cols, -1)))
        for i in range(num_rows_each):
            for j in range(self.num_components):
                rows.append(
                    tf.slice(self.g_prototypes[j].activations, (
                        num_cols * i, 0), (num_cols, -1)))
        rows = tf.concat(rows, 0)
        rows = self.ae_unnormalizer(rows)
        rows = arrange_grid(
            rows, num_rows, num_cols, self.ae_shape, transpose=True)
        self.gen_grid_all = rows

        self.add_summary(
            'gen_grid_all', rows, tf.summary.image, summary_list=summary_list)

        # -- for sampling start --

        rows = []
        rows.append(
            tf.slice(self.val_data_pos_ae_target, (0, 0), (num_cols, -1)))
        rows.append(
            tf.slice(self.g_target.activations, (0, 0), (num_cols *
                                                         num_rows_each, -1)))
        rows = tf.concat(rows, 0)
        rows = self.ae_unnormalizer(rows)
        rows = arrange_grid(
            rows, 1 + num_rows_each, num_cols, self.ae_shape, transpose=True)
        self.gen_grid_all_target = rows

        rows = []
        rows.append(
            tf.slice(self.val_data_pos_ae_source, (0, 0), (num_cols, -1)))
        rows.append(
            tf.slice(self.val_data_pos_ae_target, (0, 0), (num_cols, -1)))

        gen_grid_all_prototype = []
        for i in range(self.num_components):
            rows = rows[:2]
            rows.append(
                tf.slice(self.g_prototypes[i].activations, (0, 0), (
                    num_cols * num_rows_each, -1)))
            outputs = tf.concat(rows, 0)
            outputs = tf.concat(outputs, 0)
            outputs = self.ae_unnormalizer(outputs)
            outputs = arrange_grid(
                outputs,
                2 + num_rows_each,
                num_cols,
                self.ae_shape,
                transpose=True)
            gen_grid_all_prototype.append(outputs)
        self.gen_grid_all_prototype = gen_grid_all_prototype

        # -- for sampling end --

        num_rows = 4
        num_cols = 10
        rows = []
        rows.append(
            tf.slice(self.data_unlabeled_ae_target, (0, 0), (num_cols * 2, -1
                                                             )))
        rows.append(tf.slice(self.g.activations, (0, 0), (num_cols * 2, -1)))
        rows = tf.concat(rows, 0)
        rows = self.ae_unnormalizer(rows)
        rows = arrange_grid(rows, num_rows, num_cols, self.ae_shape)
        self.add_summary(
            'gen_grid_recon',
            rows,
            tf.summary.image,
            summary_list=summary_list)

    def train(self,
              sess,
              data,
              start_iter,
              epochs,
              post_epochs,
              best_dir,
              best_acc_dir,
              checkpoint_dir,
              epoch_callback=None,
              post_epoch_callback=None,
              save_epochs=1,
              eval_epochs=1,
              save_iters=None,
              disable_eval=False,
              saver=None,
              best_saver=None,
              best_acc_saver=None,
              writer=None,
              check=None):
        nb_train = max(data.train.num_examples_labeled_pos,
                       data.train.num_examples_labeled_neg)
        nb_unlabeled = data.train.num_examples
        logger.warning('%d pairs / %d images', nb_train, nb_unlabeled)

        nb_batch = nb_train // self.batch_size
        logger.warning('%d batches per epoch', nb_batch)

        best_auc_path = os.path.join(best_dir, 'best_accuracy')
        best_acc_path = os.path.join(best_acc_dir, 'best_accuracy_by_th')
        stats = load_best_stats(best_auc_path)
        stats_acc = load_best_stats(best_acc_path)

        if not self.gan:
            logger.info('post epochs disabled due to disabled gan')
            post_epochs = 0

        if start_iter == 0:
            self.init(sess)

        total_epochs = epochs + post_epochs
        start_epoch = start_iter // nb_batch
        logger.warning('start epoch %d of %d', start_epoch, total_epochs)
        for e in range(start_epoch, total_epochs):
            if e == start_epoch:
                t = trange(start_iter % nb_batch, nb_batch)
            else:
                t = trange(nb_batch)
            if e < epochs:
                # epochs
                t.set_description('epoch {}'.format(e))
                for i in t:
                    summary = self.summary if i != 0 else self.all_summary
                    optims = [self.s_optim, self.update_stats]
                    if not self.use_threshold:
                        optims.append(self.th_optim)

                    (summary_str, _, (train_avg, val_avg, margin_avg, pos_avg,
                                      neg_avg)) = sess.run([
                                          summary, optims, [
                                              self.s_accuracy_avg,
                                              self.val_s_accuracy_avg,
                                              self.s_margin_adapt_avg,
                                              self.s_pos_dists_adapt_avg,
                                              self.s_neg_dists_adapt_avg,
                                          ]
                                      ])
                    if writer is not None:
                        writer.add_summary(summary_str, nb_batch * e + i)
                    if save_iters and i > 0 and i % save_iters == 0 and saver is not None:
                        saver.save(
                            sess,
                            os.path.join(checkpoint_dir, 'model'),
                            global_step=nb_batch * e + i)
                    t.set_postfix(
                        error=1. - train_avg,
                        val_error=1. - val_avg,
                        margin_avg=margin_avg,
                        pos_avg=pos_avg,
                        neg_avg=neg_avg)

                if e % eval_epochs == 0 and not disable_eval:
                    # eval val accuracy
                    val_stats = dist_eval(sess, self, self.batch_size,
                                          data.val)
                    if val_stats.auc > stats.best_auc or val_stats.accuracy > stats_acc.best_accuracy:
                        test_stats = dist_eval(sess, self, self.batch_size,
                                               data.test)
                        logger.warning(('epoch %d:'
                                        ' current error ='
                                        ' train: %f val: %f test: %f'
                                        ' / auc = val: %f test: %f'), e,
                                       1. - train_avg, 1. - val_stats.accuracy,
                                       1. - test_stats.accuracy, val_stats.auc,
                                       test_stats.auc)

                        if val_stats.auc > stats.best_auc:
                            stats.best_accuracy = val_stats.accuracy
                            stats.best_auc = val_stats.auc
                            stats.best_epoch = e

                            best_saver.save(
                                sess,
                                os.path.join(best_dir, 'model'),
                                global_step=stats.best_epoch)
                            with open(best_auc_path, 'w') as outfile:
                                outfile.write('{}\t{}\t{}'.format(
                                    stats.best_epoch, stats.best_accuracy,
                                    stats.best_auc))

                        if val_stats.accuracy > stats_acc.best_accuracy:
                            stats_acc.best_accuracy = val_stats.accuracy
                            stats_acc.best_auc = val_stats.auc
                            stats_acc.best_epoch = e

                            best_acc_saver.save(
                                sess,
                                os.path.join(best_acc_dir, 'model'),
                                global_step=stats.best_epoch)
                            with open(best_acc_path, 'w') as outfile:
                                outfile.write('{}\t{}\t{}'.format(
                                    stats.best_epoch, stats.best_accuracy,
                                    stats.best_auc))

                    else:
                        logger.warning(('epoch %d:'
                                        ' current error ='
                                        ' train: %f val: %f'
                                        ' / auc = val: %f'), e, 1. - train_avg,
                                       1. - val_stats.accuracy, val_stats.auc)
                else:
                    logger.warning(('epoch %d:'
                                    ' avg error ='
                                    ' train: %f val: %f'), e, 1. - train_avg,
                                   1. - val_avg)
            else:
                # post epochs
                t.set_description('post epoch {}'.format(e))

                for i in t:
                    summary = self.post_summary if i != 0 else self.post_all_summary
                    if e == start_epoch and i == start_iter % nb_batch:
                        summary = self.post_all_summary
                    optims = []
                    optims.append(self.post_d_optim)
                    optims.append(self.post_g_optim)
                    (summary_str, _) = sess.run([
                        summary,
                        optims,
                    ])
                    if writer is not None:
                        writer.add_summary(summary_str, nb_batch * e + i)
                    if save_iters and i > 0 and i % save_iters == 0 and saver is not None:
                        saver.save(
                            sess,
                            os.path.join(checkpoint_dir, 'model'),
                            global_step=nb_batch * e + i)

            if e % save_epochs == 0 and saver is not None:
                saver.save(
                    sess,
                    os.path.join(checkpoint_dir, 'model'),
                    global_step=(e + 1) * nb_batch)
            gc.collect()


def construct_model(
        is_double, disable_double, source_shape, input_shape, ae_shape,
        pos_weight, latent_shape, batch_size, data_norm, data_type, model_type,
        gan_type, num_components, latent_size, caffe_margin, gan, cgan, t_dim,
        dist_type, act_type, use_threshold, lr, beta1, beta2, z_dim, z_stddev,
        g_dim, g_lr, g_beta1, g_beta2, m_prj, m_enc, d_dim, d_lr, d_beta1,
        d_beta2, lambda_gp, lambda_dra, lambda_m, directed, data_directed,
        reg_const, data, run_tag, train_data_transformer, val_data_transformer,
        ae_transformer, data_normalizer, data_unnormalizer, ae_normalizer,
        ae_unnormalizer, latent_normalizer, enable_input_producer):

    batches = None
    val_batches = None
    unlabeled_batches = None
    aux = None

    if enable_input_producer:
        source_shape = source_shape if source_shape else input_shape
        source_size = reduce_product(source_shape)
        if is_double:
            data_latent_size = reduce_product(latent_shape)
            num_inputs = 4

            # -- train queues --
            queues = []
            for name in ('queue_source_pos', 'queue_source_neg',
                         'queue_target_pos', 'queue_target_neg'):
                queues.append(
                    tf.placeholder(
                        tf.float32, shape=[batch_size, source_size],
                        name=name))
                queues.append(
                    tf.placeholder(
                        tf.float32,
                        shape=[batch_size, data_latent_size],
                        name=name + '_latent'))
            queue = tf.FIFOQueue(
                capacity=batch_size * 100,
                dtypes=[tf.float32, tf.float32] * num_inputs,
                shapes=[[source_size], [data_latent_size]] * num_inputs)
        else:
            num_inputs = 4

            # -- train queues --
            queues = [
                tf.placeholder(
                    tf.float32, shape=[batch_size, source_size], name=name)
                for name in ('queue_source_pos', 'queue_source_neg',
                             'queue_target_pos', 'queue_target_neg')
            ]

            queue = tf.FIFOQueue(
                capacity=batch_size * 100,
                dtypes=[tf.float32] * num_inputs,
                shapes=[[source_size] for _ in range(num_inputs)])

        enqueue_op = queue.enqueue_many(queues)
        dequeue_op = queue.dequeue()

        batches = tf.train.batch(
            dequeue_op,
            batch_size=batch_size,
            capacity=batch_size * 100,
            num_threads=2)

        # -- unlabeled queues --

        unlabeled_queues = []
        unlabeled_dtypes = []
        unlabeled_shapes = []

        if directed or data_directed:
            for party in ('src', 'dst'):
                queue_unlabeled = tf.placeholder(
                    tf.float32,
                    shape=[batch_size, source_size],
                    name='queue_unlabeled_' + party)

                unlabeled_queues.append(queue_unlabeled)
                unlabeled_dtypes.append(tf.float32)
                unlabeled_shapes.append([source_size])

                if is_double:
                    queue_unlabeled = tf.placeholder(
                        tf.float32,
                        shape=[batch_size, data_latent_size],
                        name='queue_unlabeled_' + party + '_latent')

                    unlabeled_queues.append(queue_unlabeled)
                    unlabeled_dtypes.append(tf.float32)
                    unlabeled_shapes.append([data_latent_size])

        else:
            queue_unlabeled = tf.placeholder(
                tf.float32,
                shape=[batch_size, source_size],
                name='queue_unlabeled')

            unlabeled_queues.append(queue_unlabeled)
            unlabeled_dtypes.append(tf.float32)
            unlabeled_shapes.append([source_size])

            if is_double:
                queue_unlabeled = tf.placeholder(
                    tf.float32,
                    shape=[batch_size, data_latent_size],
                    name='queue_unlabeled_latent')

                unlabeled_queues.append(queue_unlabeled)
                unlabeled_dtypes.append(tf.float32)
                unlabeled_shapes.append([data_latent_size])

            unlabeled_queues *= 2
            unlabeled_dtypes *= 2
            unlabeled_shapes *= 2

        unlabeled_queue = tf.FIFOQueue(
            capacity=batch_size * 100,
            dtypes=unlabeled_dtypes,
            shapes=unlabeled_shapes)

        unlabeled_enqueue_op = unlabeled_queue.enqueue_many(unlabeled_queues)

        unlabeled_dequeue_op = unlabeled_queue.dequeue()

        unlabeled_batches = tf.train.batch(
            unlabeled_dequeue_op,
            batch_size=batch_size,
            capacity=batch_size * 100,
            num_threads=2)

        if is_double:
            assert len(unlabeled_batches) == 4
            unlabeled_batches = [
                unlabeled_batches[0], unlabeled_batches[1], None,
                unlabeled_batches[2], unlabeled_batches[3], None
            ]
        else:
            assert len(unlabeled_batches) == 2
            unlabeled_batches = [
                unlabeled_batches[0], None, unlabeled_batches[1], None
            ]

        # -- val queues --
        if is_double:
            val_queues = []
            for name in ('val_queue_source_pos', 'val_queue_source_neg',
                         'val_queue_target_pos', 'val_queue_target_neg'):
                val_queues.append(
                    tf.placeholder(
                        tf.float32, shape=[batch_size, source_size],
                        name=name))
                val_queues.append(
                    tf.placeholder(
                        tf.float32,
                        shape=[batch_size, data_latent_size],
                        name=name + '_latent'))
            val_queue = tf.FIFOQueue(
                capacity=batch_size * 100,
                dtypes=[tf.float32, tf.float32] * num_inputs,
                shapes=[[source_size], [data_latent_size]] * num_inputs)
        else:
            val_queues = [
                tf.placeholder(
                    tf.float32, shape=[batch_size, source_size], name=name)
                for name in ('val_queue_source_pos', 'val_queue_source_neg',
                             'val_queue_target_pos', 'val_queue_target_neg')
            ]

            val_queue = tf.FIFOQueue(
                capacity=batch_size * 100,
                dtypes=[tf.float32] * num_inputs,
                shapes=[[source_size] for _ in range(num_inputs)])

        val_enqueue_op = val_queue.enqueue_many(val_queues)
        val_dequeue_op = val_queue.dequeue()

        val_batches = tf.train.batch(
            val_dequeue_op,
            batch_size=batch_size,
            capacity=batch_size * 100,
            num_threads=2)

        aux = [[enqueue_op, queues, data.train],
               [unlabeled_enqueue_op, unlabeled_queues,
                data.train], [val_enqueue_op, val_queues, data.val]]

    model = CFL(is_double=is_double,
                disable_double=disable_double,
                latent_shape=latent_shape,
                source_shape=source_shape,
                input_shape=input_shape,
                ae_shape=ae_shape,
                batch_size=batch_size,
                data_norm=data_norm,
                data_type=data_type,
                gan_type=gan_type,
                model_type=model_type,
                num_components=num_components,
                latent_size=latent_size,
                pos_weight=pos_weight,
                caffe_margin=caffe_margin,
                gan=gan,
                cgan=cgan,
                t_dim=t_dim,
                dist_type=dist_type,
                act_type=act_type,
                use_threshold=use_threshold,
                lr=lr,
                beta1=beta1,
                beta2=beta2,
                z_dim=z_dim,
                z_stddev=z_stddev,
                g_dim=g_dim,
                g_lr=g_lr,
                g_beta1=g_beta1,
                g_beta2=g_beta2,
                m_prj=m_prj,
                m_enc=m_enc,
                d_dim=d_dim,
                d_lr=d_lr,
                d_beta1=d_beta1,
                d_beta2=d_beta2,
                lambda_dra=lambda_dra,
                lambda_gp=lambda_gp,
                lambda_m=lambda_m,
                directed=directed,
                data_directed=data_directed,
                reg_const=reg_const,
                batches=batches,
                val_batches=val_batches,
                unlabeled_batches=unlabeled_batches,
                train_data_transformer=train_data_transformer,
                val_data_transformer=val_data_transformer,
                ae_transformer=ae_transformer,
                data_normalizer=data_normalizer,
                data_unnormalizer=data_unnormalizer,
                ae_normalizer=ae_normalizer,
                ae_unnormalizer=ae_unnormalizer,
                latent_normalizer=latent_normalizer,
                run_tag=run_tag)

    return model, aux
