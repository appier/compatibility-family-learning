"""Compatibility Family Learning for Item Recommendation and Generation
"""
import tensorflow as tf

from ..ops import build_placeholder
from ..ops import lrelu
from ..utils import reduce_product
from .base import ModelBase
from .blocks import Thresholder


class FCEncoder(ModelBase):
    """FCEncoder

    Encode vectors to a latent space and
    produce `num_components` prototype vectors for compatible items.
    """

    def __init__(self,
                 X,
                 input_shape,
                 num_components,
                 num_outputs,
                 batch_size,
                 initializer=tf.contrib.layers.xavier_initializer(),
                 regularizer=None,
                 layer_activation_fn=lrelu,
                 activation_fn=None,
                 name='Encoder',
                 reuse=False):
        self.input_shape = input_shape
        self.num_components = num_components
        self.num_outputs = num_outputs
        self.batch_size = batch_size
        self.regularizer = regularizer
        self.initializer = initializer

        with tf.variable_scope(name, reuse=reuse) as scope:
            super().__init__(scope)
            outputs = X

            # latent space outputs
            with tf.variable_scope('latent_outputs'):
                latent_outputs = outputs
                latent_outputs = tf.contrib.layers.fully_connected(
                    inputs=latent_outputs,
                    num_outputs=num_outputs,
                    activation_fn=activation_fn,
                    weights_regularizer=regularizer,
                    biases_regularizer=regularizer,
                    weights_initializer=initializer,
                    biases_initializer=tf.zeros_initializer())
                self.latent_outputs = latent_outputs

            # get pcd outputs
            with tf.variable_scope('pcd_outputs'):
                pcd_outputs = outputs
                pcd_outputs = tf.contrib.layers.fully_connected(
                    inputs=pcd_outputs,
                    num_outputs=num_outputs * num_components,
                    activation_fn=activation_fn,
                    weights_regularizer=regularizer,
                    weights_initializer=initializer,
                    biases_regularizer=regularizer,
                    biases_initializer=tf.zeros_initializer())
                pcd_outputs = tf.reshape(
                    pcd_outputs, [-1, self.num_components, num_outputs])
                self.pcd_outputs = pcd_outputs

    def build_dist(self, target):
        v = target.latent_outputs
        with tf.variable_scope(self.scope, reuse=True):
            if self.num_components > 1:
                diff = tf.subtract(
                    tf.reshape(v, (-1, 1, self.num_outputs)), self.pcd_outputs)
                logits = -tf.reduce_sum(tf.square(diff), axis=-1)
                scales = tf.nn.softmax(logits)
                means = tf.reduce_sum(
                    self.pcd_outputs *
                    tf.reshape(scales, (-1, self.num_components, 1)),
                    axis=-2)
                dist = tf.reduce_sum(tf.square(v - means), -1)
                return tf.reshape(dist, (-1, 1))
            else:
                diff = tf.subtract(v,
                                   tf.reshape(self.pcd_outputs,
                                              (-1, self.num_outputs)))
                dist = tf.reduce_sum(tf.square(diff), -1)
                return tf.reshape(dist, (-1, 1))


class Dist(ModelBase):
    def __init__(self,
                 input_shape,
                 latent_size,
                 num_components,
                 batch_size,
                 lr,
                 beta1,
                 beta2,
                 batches=None,
                 val_batches=None,
                 normalize_value=None,
                 data_normalizer=None,
                 data_unnormalizer=None,
                 reg_const=0.0,
                 name='Dist',
                 run_tag=None,
                 reuse=False):
        self.is_double = False
        self.input_shape = input_shape
        self.input_size = reduce_product(input_shape)
        self.latent_size = latent_size
        self.reg_const = reg_const
        self.batch_size = batch_size
        self.num_components = num_components
        self.normalize_value = normalize_value
        self.run_tag = run_tag

        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2

        self.data_normalizer = data_normalizer
        self.data_unnormalizer = data_unnormalizer

        with tf.variable_scope(name, reuse=reuse) as scope:
            super().__init__(scope)

            self.regularizer = tf.contrib.layers.l2_regularizer(
                self.reg_const) if self.reg_const > 0.0 else None

            # train data input
            input_batches = [None, None, None, None]
            if batches:
                input_batches = batches

            self.input_data_pos_source = build_placeholder(
                [None,
                 self.input_size], 'input_data_pos_source', input_batches[0])
            self.input_data_pos_target = build_placeholder(
                [None,
                 self.input_size], 'input_data_pos_target', input_batches[1])
            self.input_data_neg_source = build_placeholder(
                [None,
                 self.input_size], 'input_data_neg_source', input_batches[2])
            self.input_data_neg_target = build_placeholder(
                [None,
                 self.input_size], 'input_data_neg_target', input_batches[3])

            self.data_pos_source = data_normalizer(
                self.input_data_pos_source) if data_normalizer else None
            self.data_neg_source = data_normalizer(
                self.input_data_neg_source) if data_normalizer else None
            self.data_pos_target = data_normalizer(
                self.input_data_pos_target) if data_normalizer else None
            self.data_neg_target = data_normalizer(
                self.input_data_neg_target) if data_normalizer else None

            # val data input
            # could be used to validate the model
            val_input_batches = [None, None, None, None]
            if val_batches:
                val_input_batches = val_batches
            self.val_input_data_pos_source = build_placeholder([
                None, self.input_size
            ], 'val_input_data_pos_source', val_input_batches[0])
            self.val_input_data_pos_target = build_placeholder([
                None, self.input_size
            ], 'val_input_data_pos_target', val_input_batches[1])
            self.val_input_data_neg_source = build_placeholder([
                None, self.input_size
            ], 'val_input_data_neg_source', val_input_batches[2])
            self.val_input_data_neg_target = build_placeholder([
                None, self.input_size
            ], 'val_input_data_neg_target', val_input_batches[3])

            self.val_data_pos_source = data_normalizer(
                self.val_input_data_pos_source) if data_normalizer else None
            self.val_data_neg_source = data_normalizer(
                self.val_input_data_neg_source) if data_normalizer else None
            self.val_data_pos_target = data_normalizer(
                self.val_input_data_pos_target) if data_normalizer else None
            self.val_data_neg_target = data_normalizer(
                self.val_input_data_neg_target) if data_normalizer else None

            self._build_model(reuse=reuse)
            self._build_losses()
            self._build_optimizer()
            self._build_summary()

    def get_name(self):
        name = 'linear_dist'
        name += '_ls_{}_nc_{}_reg_{}_norm_{}'.format(
            self.latent_size, self.num_components, self.reg_const,
            self.normalize_value)
        if self.run_tag:
            name += '_run_' + self.run_tag
        return name

    def dist_fn(self, inputs, reuse=False):
        return FCEncoder(
            inputs,
            input_shape=self.input_shape,
            num_components=self.num_components,
            num_outputs=self.latent_size,
            batch_size=self.batch_size,
            regularizer=self.regularizer,
            reuse=reuse)

    def thres_fn(self, inputs, reuse=False):
        return Thresholder(inputs, reuse=reuse)

    def _build_model(self, reuse):
        self.s_pos_src = self.dist_fn(self.data_pos_source)

        self.s_pos_target = self.dist_fn(self.data_pos_target, reuse=True)

        self.s_pos_dists = self.s_pos_src.build_dist(self.s_pos_target)

        self.s_pos_predicts = self.thres_fn(self.s_pos_dists)

        self.s_neg_src = self.dist_fn(self.data_neg_source, reuse=True)
        self.s_neg_target = self.dist_fn(self.data_neg_target, reuse=True)

        self.s_neg_dists = self.s_neg_src.build_dist(self.s_neg_target)

        self.s_neg_predicts = self.thres_fn(self.s_neg_dists, reuse=True)

        # -- val --

        self.val_s_pos_src = self.dist_fn(self.val_data_pos_source, reuse=True)

        self.val_s_pos_target = self.dist_fn(
            self.val_data_pos_target, reuse=True)

        self.val_s_pos_dists = self.val_s_pos_src.build_dist(
            self.val_s_pos_target)

        self.val_s_pos_predicts = self.thres_fn(
            self.val_s_pos_dists, reuse=True)

        self.val_s_neg_src = self.dist_fn(self.val_data_neg_source, reuse=True)
        self.val_s_neg_target = self.dist_fn(
            self.val_data_neg_target, reuse=True)

        self.val_s_neg_dists = self.val_s_neg_src.build_dist(
            self.val_s_neg_target)

        self.val_s_neg_predicts = self.thres_fn(
            self.val_s_neg_dists, reuse=True)

    def _build_losses(self):
        # regularizations

        self.s_loss_reg = self.s_pos_src.reg_loss()
        self.s_total_loss = self.s_loss_reg

        self.s_margins = tf.reduce_mean(self.s_pos_dists - self.s_neg_dists)
        self.s_p_loss_pos = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                logits=self.s_pos_predicts.outputs,
                labels=tf.ones_like(self.s_pos_predicts.outputs)))
        self.s_p_loss_neg = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                logits=self.s_neg_predicts.outputs,
                labels=tf.zeros_like(self.s_neg_predicts.outputs)))
        self.thres_loss = self.s_p_loss_pos + self.s_p_loss_neg
        self.s_total_loss += self.thres_loss

        correct_prediction_pos = tf.greater(self.s_pos_predicts.outputs, 0.0)
        correct_prediction_neg = tf.less_equal(self.s_neg_predicts.outputs,
                                               0.0)
        self.s_accuracy = (
            tf.reduce_mean(tf.cast(correct_prediction_pos, tf.float32)) +
            tf.reduce_mean(tf.cast(correct_prediction_neg, tf.float32))) / 2.0

        correct_prediction_pos = tf.greater(self.val_s_pos_predicts.outputs,
                                            0.0)
        correct_prediction_neg = tf.less_equal(self.val_s_neg_predicts.outputs,
                                               0.0)
        self.val_s_accuracy = (
            tf.reduce_mean(tf.cast(correct_prediction_pos, tf.float32)) +
            tf.reduce_mean(tf.cast(correct_prediction_neg, tf.float32))) / 2.0

    def _build_optimizer(self):
        self.s_vars = self.s_neg_src.get_vars()
        th_vars = self.s_pos_predicts.get_vars()
        self.s_vars.extend(th_vars)

        self.s_optim = tf.train.AdamOptimizer(
            self.lr, beta1=self.beta1, beta2=self.beta2).minimize(
                self.s_total_loss, var_list=self.s_vars)

    def _build_summary(self):
        with tf.name_scope('summary') as scope:
            self.thres_loss_sum = tf.summary.scalar('thres_loss',
                                                    self.thres_loss)
            self.thres_sum = tf.summary.scalar('threshold',
                                               self.s_neg_predicts.threshold)
            self.s_accuracy_sum = tf.summary.scalar('s_accuracy',
                                                    self.s_accuracy)
            self.val_s_accuracy_sum = tf.summary.scalar('val_s_accuracy',
                                                        self.val_s_accuracy)
            self.s_pos_predicts_sum = tf.summary.scalar(
                's_pos_predicts', tf.reduce_mean(self.s_pos_predicts.outputs))
            self.s_neg_predicts_sum = tf.summary.scalar(
                's_neg_predicts', tf.reduce_mean(self.s_neg_predicts.outputs))
            self.s_p_pos_loss_sum = tf.summary.scalar('s_p_pos_loss',
                                                      self.s_p_loss_pos)
            self.s_p_neg_loss_sum = tf.summary.scalar('s_p_neg_loss',
                                                      self.s_p_loss_neg)

            self.s_margins_sum = tf.summary.scalar('s_margins', self.s_margins)
            self.s_pos_dists_sum = tf.summary.scalar(
                's_pos_dists', tf.reduce_mean(self.s_pos_dists))
            self.s_neg_dists_sum = tf.summary.scalar(
                's_neg_dists', tf.reduce_mean(self.s_neg_dists))
            self.s_total_loss_sum = tf.summary.scalar('s_total_loss',
                                                      self.s_total_loss)
            self.val_s_pos_dists_sum = tf.summary.scalar(
                'val_s_pos_dists', tf.reduce_mean(self.val_s_pos_dists))
            self.val_s_neg_dists_sum = tf.summary.scalar(
                'val_s_neg_dists', tf.reduce_mean(self.val_s_neg_dists))
            summaries = tf.get_collection(tf.GraphKeys.SUMMARIES, scope=scope)
            self.summary = tf.summary.merge(summaries)
            self.all_summary = self.summary


def construct_model(input_shape,
                    latent_size,
                    num_components,
                    lr,
                    beta1,
                    beta2,
                    batch_size,
                    normalize_value,
                    reg_const=0.0,
                    data_normalizer=None,
                    data_unnormalizer=None,
                    data=None,
                    run_tag=None):
    if data is None:
        model = Dist(
            input_shape=input_shape,
            latent_size=latent_size,
            num_components=num_components,
            reg_const=reg_const,
            batch_size=batch_size,
            lr=lr,
            beta1=beta1,
            beta2=beta2,
            batches=None,
            val_batches=None,
            normalize_value=normalize_value,
            data_normalizer=data_normalizer,
            data_unnormalizer=data_unnormalizer,
            run_tag=run_tag)

        aux = None
        return model, aux

    else:
        input_size = reduce_product(input_shape)
        num_inputs = 4

        queue_pos_source = tf.placeholder(
            tf.float32,
            shape=[batch_size, input_size],
            name='queue_source_pos')
        queue_neg_source = tf.placeholder(
            tf.float32,
            shape=[batch_size, input_size],
            name='queue_source_neg')
        queue_pos_target = tf.placeholder(
            tf.float32,
            shape=[batch_size, input_size],
            name='queue_target_pos')
        queue_neg_target = tf.placeholder(
            tf.float32,
            shape=[batch_size, input_size],
            name='queue_target_neg')

        queue = tf.FIFOQueue(
            capacity=batch_size * 100,
            dtypes=[tf.float32] * num_inputs,
            shapes=[[input_size] for _ in range(num_inputs)])

        queues = [
            queue_pos_source, queue_pos_target, queue_neg_source,
            queue_neg_target
        ]
        enqueue_op = queue.enqueue_many(queues)
        dequeue_op = queue.dequeue()

        batches = tf.train.batch(
            dequeue_op,
            batch_size=batch_size,
            capacity=batch_size * 100,
            num_threads=2)

        # -- val data --

        val_queue_pos_source = tf.placeholder(
            tf.float32,
            shape=[batch_size, input_size],
            name='val_queue_source_pos')
        val_queue_neg_source = tf.placeholder(
            tf.float32,
            shape=[batch_size, input_size],
            name='val_queue_source_neg')
        val_queue_pos_target = tf.placeholder(
            tf.float32,
            shape=[batch_size, input_size],
            name='val_queue_target_pos')
        val_queue_neg_target = tf.placeholder(
            tf.float32,
            shape=[batch_size, input_size],
            name='val_queue_target_neg')

        val_queue = tf.FIFOQueue(
            capacity=batch_size * 100,
            dtypes=[tf.float32] * num_inputs,
            shapes=[[input_size] for _ in range(num_inputs)])

        val_queues = [
            val_queue_pos_source, val_queue_pos_target, val_queue_neg_source,
            val_queue_neg_target
        ]

        val_enqueue_op = val_queue.enqueue_many(val_queues)
        val_dequeue_op = val_queue.dequeue()

        val_batches = tf.train.batch(
            val_dequeue_op,
            batch_size=batch_size,
            capacity=batch_size * 100,
            num_threads=2)

        model = Dist(
            input_shape=input_shape,
            latent_size=latent_size,
            num_components=num_components,
            lr=lr,
            beta1=beta1,
            beta2=beta2,
            reg_const=reg_const,
            batch_size=batch_size,
            batches=batches,
            val_batches=val_batches,
            normalize_value=normalize_value,
            data_normalizer=data_normalizer,
            data_unnormalizer=data_unnormalizer,
            run_tag=run_tag)

        aux = [[enqueue_op, queues, data.train],
               [val_enqueue_op, val_queues, data.val]]

        return model, aux
