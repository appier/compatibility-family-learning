import argparse
import logging
import os
import re

from argparse import Namespace

import numpy as np
import tensorflow as tf

from sklearn.metrics import roc_auc_score, roc_curve
from tqdm import tqdm
from scipy.misc import imsave

from .ops import np_arrange_grid

logger = logging.getLogger(__name__)


def log_args(args):
    for name, value in sorted(vars(args).items()):
        logger.warning('%s = %r', name, value)


def reduce_product(xs):
    s = 1
    for x in xs:
        s *= x
    return s


def load_best_stats(path):
    stats = Namespace(best_epoch=None, best_accuracy=0.0, best_auc=0.0)
    if os.path.exists(path):
        with open(path) as infile:
            tokens = infile.read().split('\t')
            best_epoch, best_accuracy = tokens[:2]
            stats.best_epoch = int(best_epoch)
            stats.best_accuracy = float(best_accuracy)
            if len(tokens) >= 3:
                stats.best_auc = float(tokens[2])
    return stats


def dist_check_args(args):
    if args.data_is_double:
        assert args.data_is_image
        assert args.latent_shape is not None
        args.latent_shape = tuple(args.latent_shape)

    if args.input_shape:
        args.input_shape = tuple(args.input_shape)

    if args.ae_shape:
        args.ae_shape = tuple(args.ae_shape)

    if args.source_shape:
        args.source_shape = tuple(args.source_shape)

    if args.data_mean:
        args.data_mean = tuple(args.data_mean)

    if args.data_norm:
        args.data_norm = tuple(args.data_norm)

    if args.caffe_margin and args.caffe_margin > 0:
        assert args.dist_type == 'siamese', 'only use cd loss in siamese'

    if args.caffe_margin and args.caffe_margin < 0:
        assert args.lambda_m > 0

    if args.lambda_m > 0:
        assert not args.caffe_margin

    if args.dist_type != 'siamese':
        assert args.use_threshold, 'must use entropy loss'

    if not args.cgan:
        assert not args.t_dim

    log_args(args)


def dist_parser(data_name='mnist',
                data_root='parsed_data',
                checkpoint_root='checkpoints',
                log_root='logs',
                run_tag=None,
                seed=633,
                source_shape=None,
                input_shape=(28, 28, 1),
                ae_shape=None,
                batch_size=100,
                data_scale=None,
                data_mean=None,
                data_norm=None,
                latent_norm=None,
                data_type='sigmoid',
                model_type='conv',
                dist_type='pcd',
                act_type=None,
                num_components=2,
                latent_size=20,
                pos_weight=None,
                lr=0.001,
                beta1=0.9,
                beta2=0.999,
                z_dim=20,
                z_stddev=1.,
                g_dim=64,
                g_lr=0.0002,
                g_beta1=0.5,
                g_beta2=0.999,
                m_prj=None,
                m_enc=None,
                gan_type='conv',
                caffe_margin=None,
                d_dim=64,
                t_dim=None,
                d_lr=0.0002,
                d_beta1=0.5,
                d_beta2=0.999,
                lambda_dra=0.5,
                lambda_gp=None,
                lambda_m=0.0,
                reg_const=0.0):
    parser = argparse.ArgumentParser()
    # -- run params --
    parser.add_argument('--data-name', default=data_name)
    parser.add_argument('--data-root', default=data_root)
    parser.add_argument('--checkpoint-root', default=checkpoint_root)
    parser.add_argument('--log-root', default=log_root)
    parser.add_argument('--run-tag', default=run_tag)
    parser.add_argument('--seed', type=int, default=seed)

    # -- data params --
    # (data * data_scale - mean) / normalize_value
    parser.add_argument(
        '--source-shape',
        nargs='+',
        type=int,
        help='shape of input source',
        default=source_shape)
    parser.add_argument(
        '--input-shape',
        nargs='+',
        type=int,
        help='shape to crop for model',
        default=input_shape)
    parser.add_argument(
        '--ae-shape',
        nargs='+',
        type=int,
        help='shape for autoencoder / generator',
        default=ae_shape)
    parser.add_argument('--batch-size', type=int, default=batch_size)
    parser.add_argument('--data-scale', type=float, default=data_scale)
    parser.add_argument(
        '--data-mean', nargs='+', type=float, default=data_mean)
    parser.add_argument(
        '--data-norm', nargs='+', type=float, default=data_norm)
    parser.add_argument(
        '--data-type',
        default=data_type,
        choices=('sigmoid', 'tanh', 'relu', 'linear'),
        help='range of input data; enforce clip as well')
    parser.add_argument('--data-mirror', action='store_true')
    parser.add_argument('--data-random-crop', action='store_true')
    parser.add_argument('--data-is-image', action='store_true')
    parser.add_argument('--latent-shape', type=int, nargs='+')
    parser.add_argument('--latent-norm', type=float, default=latent_norm)
    parser.add_argument('--data-is-double', action='store_true')
    parser.add_argument('--data-disable-double', action='store_true')
    parser.add_argument('--raw-latent', action='store_true')

    # -- model params --
    parser.add_argument('--pos-weight', type=float, default=pos_weight)
    parser.add_argument(
        '--model-type', default=model_type, choices=('linear', 'conv'))
    parser.add_argument('--num-components', type=int, default=num_components)
    parser.add_argument('--latent-size', type=int, default=latent_size)
    parser.add_argument('--lr', type=float, default=lr)
    parser.add_argument('--beta1', type=float, default=beta1)
    parser.add_argument('--beta2', type=float, default=beta2)
    parser.add_argument('--z-dim', type=int, default=z_dim)
    parser.add_argument('--z-stddev', type=float, default=z_stddev)
    parser.add_argument('--g-dim', type=int, default=g_dim)
    parser.add_argument('--g-lr', type=float, default=g_lr)
    parser.add_argument('--g-beta1', type=float, default=g_beta1)
    parser.add_argument('--g-beta2', type=float, default=g_beta2)
    parser.add_argument('--m-prj', type=float, default=m_prj)
    parser.add_argument('--m-enc', type=float, default=m_enc)
    parser.add_argument('--d-dim', type=int, default=d_dim)
    parser.add_argument('--d-lr', type=float, default=d_lr)
    parser.add_argument('--d-beta1', type=float, default=d_beta1)
    parser.add_argument('--d-beta2', type=float, default=d_beta2)
    parser.add_argument('--lambda-dra', type=float, default=lambda_dra)
    parser.add_argument('--lambda-gp', type=float, default=lambda_gp)
    parser.add_argument('--lambda-m', type=float, default=lambda_m)
    parser.add_argument('--gan-type', default=gan_type)
    parser.add_argument('--gan', action='store_true', help='enable GAN')
    parser.add_argument('--cgan', action='store_true', help='enable CGAN')
    parser.add_argument('--t-dim', type=int, default=t_dim)
    parser.add_argument(
        '--dist-type',
        help='distance type',
        choices=('monomer', 'pcd', 'siamese'),
        default=dist_type)
    parser.add_argument(
        '--act-type',
        help='act type',
        choices=('sigmoid', 'tanh', 'relu', 'linear'),
        default=act_type)
    parser.add_argument('--use-threshold', action='store_true')
    parser.add_argument(
        '--caffe-margin',
        type=float,
        default=caffe_margin,
        help='margin for CD loss for siamese training')
    parser.add_argument('--directed', action='store_true')
    parser.add_argument('--data-directed', action='store_true')
    parser.add_argument('--reg-const', type=float, default=reg_const)

    return parser


def dist_eval(sess, model, batch_size, data):
    total = 0
    correct = 0
    y_true = []
    y_score = []
    base_feed_dict = {}
    for batches in data.whole_pos_batches(batch_size):
        if model.is_double:
            feed_dict = {
                model.val_input_data_pos_source_latent: batches[1],
                model.val_input_data_pos_target_latent: batches[3],
            }
        else:
            feed_dict = {
                model.val_input_data_pos_source: batches[0],
                model.val_input_data_pos_target: batches[1],
            }
        feed_dict.update(base_feed_dict)
        dists = sess.run(model.val_s_pos_predicts.outputs, feed_dict=feed_dict)
        total += dists.shape[0]
        correct += (dists > 0).astype(int).sum()
        y_score.extend(dists)
        y_true.extend([1] * dists.shape[0])
    for batches in data.whole_neg_batches(batch_size):
        if model.is_double:
            feed_dict = {
                model.val_input_data_pos_source_latent: batches[1],
                model.val_input_data_pos_target_latent: batches[3],
            }
        else:
            feed_dict = {
                model.val_input_data_pos_source: batches[0],
                model.val_input_data_pos_target: batches[1],
            }
        feed_dict.update(base_feed_dict)
        dists = sess.run(model.val_s_pos_predicts.outputs, feed_dict=feed_dict)
        total += dists.shape[0]
        correct += (dists <= 0).astype(int).sum()
        y_score.extend(dists)
        y_true.extend([0] * dists.shape[0])
    roc = roc_curve(y_true, y_score)
    auc = roc_auc_score(y_true, y_score)
    report = Namespace(
        error=(total - correct) / total,
        accuracy=correct / total,
        auc=auc,
        roc=roc)
    return report


def dist_predict(sess, model, data, batch_size, predict_dir, output_name):
    logger.warning('predict %s...', output_name)
    pos_dists = []
    base_feed_dict = {}
    for batches in data.whole_pos_batches(batch_size):
        if model.is_double:
            feed_dict = {
                model.val_input_data_pos_source_latent: batches[1],
                model.val_input_data_pos_target_latent: batches[3],
            }
        else:
            feed_dict = {
                model.val_input_data_pos_source: batches[0],
                model.val_input_data_pos_target: batches[1],
            }
        feed_dict.update(base_feed_dict)
        dists = sess.run(model.val_s_pos_predicts.outputs, feed_dict=feed_dict)
        pos_dists.extend(dists)
    neg_dists = []
    for batches in data.whole_neg_batches(batch_size):
        if model.is_double:
            feed_dict = {
                model.val_input_data_pos_source_latent: batches[1],
                model.val_input_data_pos_target_latent: batches[3],
            }
        else:
            feed_dict = {
                model.val_input_data_pos_source: batches[0],
                model.val_input_data_pos_target: batches[1],
            }
        feed_dict.update(base_feed_dict)
        dists = sess.run(model.val_s_pos_predicts.outputs, feed_dict=feed_dict)
        neg_dists.extend(dists)

    rules = [
        (pos_dists, data.pairs_pos),
        (neg_dists, data.pairs_neg),
    ]

    os.makedirs(predict_dir, exist_ok=True)
    with open(os.path.join(predict_dir, output_name), 'w') as outfile:
        for dists, pairs in rules:
            for pred, (idx1, idx2) in zip(dists, pairs):
                outfile.write('{} match {} {}\n'.format(data.index_to_asins[
                    idx1], data.index_to_asins[idx2], pred[0]))


def dist_sample_near(sess, model, data, batch_size, sample_dir, output_name):
    os.makedirs(sample_dir, exist_ok=True)
    logger.warning('sample %s...', output_name)
    base_feed_dict = {}
    repeats = batch_size // 10
    for i, batches in enumerate(data.whole_pos_batches(10)):
        if model.is_double:
            pos_target = list(batches[2:4])
            pos_target[0] = np.tile(pos_target[0], (repeats, 1))
            pos_target[1] = np.tile(pos_target[1], (repeats, 1))
            pos_size = pos_target[0].shape[0]
        else:
            pos_target = batches[1]
            pos_target = np.tile(pos_target, (repeats, 1))
            pos_size = pos_target.shape[0]
        if pos_size == batch_size:
            if model.is_double:
                feed_dict = {
                    model.val_input_data_pos_target: pos_target[0],
                    model.val_input_data_pos_target_latent: pos_target[1],
                }
            else:
                feed_dict = {
                    model.val_input_data_pos_target: pos_target,
                }
            feed_dict.update(base_feed_dict)
            image = sess.run(model.gen_grid_all_target, feed_dict=feed_dict)[0]
            if len(image.shape) > 2 and image.shape[2] == 1:
                image = image.reshape(image.shape[:2])
            imsave(
                os.path.join(sample_dir, '{}_{:010d}.png'.format(output_name,
                                                                 i)), image)


def dist_sample(sess, model, data, batch_size, sample_dir, output_name):
    os.makedirs(sample_dir, exist_ok=True)
    logger.warning('sample %s...', output_name)
    base_feed_dict = {}
    repeats = batch_size // 10
    for i, batches in enumerate(data.whole_pos_batches(10)):
        if model.is_double:
            pos_source = list(batches[:2])
            pos_source[0] = np.tile(pos_source[0], (repeats, 1))
            pos_source[1] = np.tile(pos_source[1], (repeats, 1))
            pos_size = pos_source[0].shape[0]
        else:
            pos_source = batches[0]
            pos_source = np.tile(pos_source, (repeats, 1))
            pos_size = pos_source.shape[0]
        if pos_size == batch_size:
            if model.is_double:
                feed_dict = {
                    model.val_input_data_pos_source: pos_source[0],
                    model.val_input_data_pos_source_latent: pos_source[1],
                }
            else:
                feed_dict = {
                    model.val_input_data_pos_source: pos_source,
                }
            feed_dict.update(base_feed_dict)
            image = sess.run(model.gen_grid_all, feed_dict=feed_dict)[0]
            if len(image.shape) > 2 and image.shape[2] == 1:
                image = image.reshape(image.shape[:2])
            imsave(
                os.path.join(sample_dir, '{}_{:010d}.png'.format(output_name,
                                                                 i)), image)


def dist_sample_project_disc(sess, model, data, batch_size, sample_dir,
                             output_name):
    os.makedirs(sample_dir, exist_ok=True)
    logger.warning('sample %s...', output_name)
    base_feed_dict = {}
    for batches in data.whole_pos_batches(batch_size, source_ids=True):
        for one_item in zip(*batches):
            if model.is_double:
                pos_source_image = one_item[0]
                pos_target_image = one_item[2]

                pos_source = np.tile(one_item[0].reshape((1, -1)), (batch_size,
                                                                    1))
                pos_source_latent = np.tile(one_item[1].reshape((1, -1)),
                                            (batch_size, 1))
                pos_target = np.tile(one_item[2].reshape((1, -1)), (batch_size,
                                                                    1))
                pos_target_latent = np.tile(one_item[3].reshape((1, -1)),
                                            (batch_size, 1))

                feed_dict = {
                    model.val_input_data_pos_source: pos_source,
                    model.val_input_data_pos_source_latent: pos_source_latent,
                    model.val_input_data_pos_target: pos_target,
                    model.val_input_data_pos_target_latent: pos_target_latent,
                }
            else:
                pos_source_image = one_item[0]
                pos_target_image = one_item[1]

                pos_source = np.tile(one_item[0].reshape((1, -1)), (batch_size,
                                                                    1))
                pos_target = np.tile(one_item[1].reshape((1, -1)), (batch_size,
                                                                    1))

                feed_dict = {
                    model.val_input_data_pos_source: pos_source,
                    model.val_input_data_pos_target: pos_target,
                }
            feed_dict.update(base_feed_dict)
            d_prototypes = [
                d_prototype.disc_activations
                for d_prototype in model.d_prototypes
            ]
            g_prototypes = [
                model.ae_unnormalizer(g_prototype.activations)
                for g_prototype in model.g_prototypes
            ]
            images, d_preds = sess.run(
                [g_prototypes, d_prototypes], feed_dict=feed_dict)
            images = np.concatenate(images)
            d_preds = np.concatenate(d_preds).flatten()

            indices = np.argsort(-d_preds)
            images = images[indices]
            images = np_arrange_grid(images,
                                     batch_size // 10 * model.num_components,
                                     10, model.ae_shape)[0]

            pos_source_image = pos_source_image.reshape(model.ae_shape)
            pos_target_image = pos_target_image.reshape(model.ae_shape)

            imsave(
                os.path.join(sample_dir, '{}_{}.png'.format(
                    output_name, one_item[-1])), images)
            imsave(
                os.path.join(sample_dir, '{}_{}_src.png'.format(
                    output_name, one_item[-1])), pos_source_image)
            imsave(
                os.path.join(sample_dir, '{}_{}_dst.png'.format(
                    output_name, one_item[-1])), pos_target_image)


def load_model(sess, checkpoint_dir, load_pre_weights=None):
    saver = tf.train.Saver()

    sess.run(tf.global_variables_initializer())

    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
        checkpoint_path = ckpt.model_checkpoint_path

        saver.restore(sess, checkpoint_path)
        ckpt_name = os.path.basename(checkpoint_path)
        start_step = int(
            re.search(r'(\d+)', ckpt_name.split('-')[-1]).group(1)) + 1
        logger.info(checkpoint_path, 'loaded')
    else:
        start_step = 0
        if load_pre_weights:
            best_model = os.path.join(load_pre_weights, 'best_model')
            ckpt = tf.train.get_checkpoint_state(best_model)
            ckpt_all = tf.train.get_checkpoint_state(load_pre_weights)
            if ckpt and ckpt.model_checkpoint_path and ckpt_all and ckpt_all.model_checkpoint_path:
                checkpoint_path = ckpt.model_checkpoint_path
                tf.contrib.framework.assign_from_checkpoint_fn(
                    checkpoint_path,
                    tf.trainable_variables(),
                    ignore_missing_vars=True)(sess)
                ckpt_name = os.path.basename(ckpt_all.model_checkpoint_path)
                start_step = int(
                    re.search(r'(\d+)', ckpt_name.split('-')[-1]).group(1)) + 1
                logger.info(checkpoint_path, 'loaded')
            else:
                raise Exception('must have best model! %s', best_model)
    return saver, start_step


class IncrementalAverage(object):
    def __init__(self):
        self.average = 0.0
        self.count = 0

    def add(self, value):
        self.count += 1
        self.average = (value - self.average) / self.count + self.average


def monomer_parser(data_name='monomer/Baby-also_viewed',
                   data_root='parsed_data',
                   checkpoint_root='checkpoints',
                   log_root='logs',
                   run_tag=None,
                   seed=633,
                   input_shape=(4096, ),
                   batch_size=100,
                   normalize_value=1.0,
                   num_components=2,
                   latent_size=20,
                   lr=0.001,
                   beta1=0.9,
                   beta2=0.999,
                   reg_const=0.0):
    parser = argparse.ArgumentParser()
    # -- run params --
    parser.add_argument('--data-name', default=data_name)
    parser.add_argument('--data-root', default=data_root)
    parser.add_argument('--checkpoint-root', default=checkpoint_root)
    parser.add_argument('--log-root', default=log_root)
    parser.add_argument('--run-tag', default=run_tag)
    parser.add_argument('--seed', type=int, default=seed)

    # -- data params --
    parser.add_argument(
        '--normalize-value', type=float, default=normalize_value)
    parser.add_argument(
        '--input-shape',
        nargs='+',
        type=int,
        help='shape of input',
        default=input_shape)
    parser.add_argument('--batch-size', type=int, default=batch_size)

    # -- model params --
    parser.add_argument('--num-components', type=int, default=num_components)
    parser.add_argument('--latent-size', type=int, default=latent_size)
    parser.add_argument('--lr', type=float, default=lr)
    parser.add_argument('--beta1', type=float, default=beta1)
    parser.add_argument('--beta2', type=float, default=beta2)
    parser.add_argument('--reg-const', type=float, default=reg_const)

    return parser
