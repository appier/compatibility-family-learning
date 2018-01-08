import logging
import os
import shutil
import threading

import tensorflow as tf

from ..input_data import load_data_sets
from ..models.cfl import construct_model
from ..ops import dist_ae_transformer
from ..ops import dist_normalizer
from ..ops import dist_transformer
from ..utils import dist_check_args
from ..utils import dist_parser
from ..utils import load_model, reduce_product

logger = logging.getLogger(__name__)


def enqueue(coord,
            sess,
            aux,
            batch_size,
            unlabeled=False,
            return_labels=False,
            directed=False):
    (enqueue_op, queues, data) = aux

    while not coord.should_stop():
        if unlabeled:
            if directed:
                batch_src = data.next_source_batch(
                    batch_size, return_labels=return_labels)
                batch_dst = data.next_target_batch(
                    batch_size, return_labels=return_labels)
                batch = batch_src + batch_dst
            else:
                batch = data.next_unlabeled_batch(
                    batch_size, return_labels=return_labels)
                batch *= 2
        else:
            batch = data.next_batch(batch_size)
        assert len(queues) == len(batch), '{} != {}'.format(
            len(queues), len(batch))
        feed_dict = {queue: subbatch for queue, subbatch in zip(queues, batch)}
        sess.run(enqueue_op, feed_dict=feed_dict)


def train_loop(sess, model, data, aux, batch_size, start_iter, epochs,
               post_epochs, eval_epochs, save_iters, disable_eval, log_dir,
               checkpoint_dir, saver, directed):

    best_saver = tf.train.Saver()
    best_acc_saver = tf.train.Saver()
    writer = tf.summary.FileWriter(log_dir, sess.graph)

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord, sess=sess)

    # enqueue train
    enqueue_thread = threading.Thread(
        target=enqueue, args=[coord, sess, aux[0], batch_size])
    enqueue_thread.daemon = True
    enqueue_thread.start()

    # enqueue unlabeled
    kwargs = {'unlabeled': True, 'directed': directed}
    enqueue_thread = threading.Thread(
        target=enqueue, args=[coord, sess, aux[1], batch_size], kwargs=kwargs)
    enqueue_thread.daemon = True
    enqueue_thread.start()

    # enqueue val
    enqueue_thread = threading.Thread(
        target=enqueue, args=[coord, sess, aux[2], batch_size])
    enqueue_thread.daemon = True
    enqueue_thread.start()

    best_dir = os.path.join(checkpoint_dir, 'best_model')
    os.makedirs(best_dir, exist_ok=True)
    best_acc_dir = os.path.join(checkpoint_dir, 'best_acc_model')
    os.makedirs(best_acc_dir, exist_ok=True)
    try:
        model.train(
            sess=sess,
            data=data,
            start_iter=start_iter,
            epochs=epochs,
            post_epochs=post_epochs,
            best_dir=best_dir,
            best_acc_dir=best_acc_dir,
            checkpoint_dir=checkpoint_dir,
            eval_epochs=eval_epochs,
            disable_eval=disable_eval,
            saver=saver,
            best_saver=best_saver,
            best_acc_saver=best_acc_saver,
            save_iters=save_iters,
            writer=writer)
    finally:
        coord.request_stop()
        coord.join(threads)


def train_dist(load_pre_weights, data_name, data_root, checkpoint_root,
               data_switch, log_root, run_tag, seed, source_shape, input_shape,
               ae_shape, pos_weight, batch_size, data_scale, data_mean,
               data_norm, data_type, data_mirror, data_random_crop,
               data_is_image, data_is_double, data_disable_double, latent_norm,
               latent_shape, model_type, gan_type, num_components, latent_size,
               raw_latent, caffe_margin, gan, cgan, t_dim, dist_type, act_type,
               use_threshold, lr, beta1, beta2, z_dim, z_stddev, g_dim, g_lr,
               g_beta1, g_beta2, m_prj, m_enc, d_dim, d_lr, d_beta1, d_beta2,
               lambda_dra, lambda_gp, lambda_m, directed, data_directed,
               reg_const, epochs, post_epochs, eval_epochs, save_iters,
               disable_eval, reset):
    tf.set_random_seed(seed)

    input_size = reduce_product(input_shape)
    source_size = reduce_product(source_shape) if source_shape else input_size

    data_dir = os.path.join(data_root, data_name)
    checkpoint_dir = os.path.join(checkpoint_root, data_name)
    log_dir = os.path.join(log_root, data_name)

    # load data sets
    data = load_data_sets(
        data_dir,
        source_size,
        is_image=data_is_image,
        is_double=data_is_double,
        raw_latent=raw_latent,
        directed=directed or data_directed,
        data_switch=data_switch,
        seed=seed)

    train_data_transformer, val_data_transformer = dist_transformer(
        source_shape=source_shape,
        input_shape=input_shape,
        data_random_crop=data_random_crop,
        data_mirror=data_mirror)

    ae_transformer = dist_ae_transformer(
        input_shape=input_shape, ae_shape=ae_shape)

    (data_normalizer, data_unnormalizer, ae_normalizer, ae_unnormalizer,
     latent_normalizer) = dist_normalizer(
         input_shape=input_shape,
         ae_shape=ae_shape,
         data_scale=data_scale,
         data_mean=data_mean,
         data_norm=data_norm,
         latent_norm=latent_norm,
         data_type=data_type)

    model, aux = construct_model(
        is_double=data_is_double,
        disable_double=data_disable_double,
        latent_shape=latent_shape,
        source_shape=source_shape,
        input_shape=input_shape,
        ae_shape=ae_shape,
        batch_size=batch_size,
        data_norm=data_norm,
        data_type=data_type,
        model_type=model_type,
        gan_type=gan_type,
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
        data=data,
        run_tag=run_tag,
        train_data_transformer=train_data_transformer,
        val_data_transformer=val_data_transformer,
        ae_transformer=ae_transformer,
        data_normalizer=data_normalizer,
        data_unnormalizer=data_unnormalizer,
        ae_normalizer=ae_normalizer,
        ae_unnormalizer=ae_unnormalizer,
        latent_normalizer=latent_normalizer,
        enable_input_producer=True)

    # get paths
    no_gan_checkpoint_dir = os.path.join(
        checkpoint_dir,
        model.get_name(no_gan=True)) if load_pre_weights else None
    checkpoint_dir = os.path.join(checkpoint_dir, model.get_name())
    log_dir = os.path.join(log_dir, model.get_name())

    # reset files
    if reset:
        for path in [checkpoint_dir, log_dir]:
            if os.path.exists(path):
                shutil.rmtree(path)

    # create files
    for path in [checkpoint_dir, log_dir]:
        os.makedirs(path, exist_ok=True)

    # save logging to text file
    log_format = '%(asctime)s [%(levelname)-5.5s] [%(name)s]  %(message)s'
    logging.basicConfig(
        filename=os.path.join(log_dir, 'log.log'),
        format=log_format,
        level=logging.WARNING)

    rootLogger = logging.getLogger()
    formatter = logging.Formatter(log_format)

    consoleHandler = logging.StreamHandler()
    consoleHandler.setLevel(logging.INFO)
    consoleHandler.setFormatter(formatter)
    rootLogger.addHandler(consoleHandler)
    logger.warning('run with %s', model.get_name())

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    logger.info('start session')
    with tf.Session(config=config) as sess:
        saver, start_iter = load_model(sess, checkpoint_dir,
                                       no_gan_checkpoint_dir)
        train_loop(
            sess=sess,
            model=model,
            aux=aux,
            data=data,
            batch_size=batch_size,
            start_iter=start_iter,
            epochs=epochs,
            post_epochs=post_epochs,
            eval_epochs=eval_epochs,
            log_dir=log_dir,
            checkpoint_dir=checkpoint_dir,
            disable_eval=disable_eval,
            save_iters=save_iters,
            saver=saver,
            directed=directed or data_directed)


def main():
    args = parse_args()
    train_dist(**vars(args))


def parse_args():
    parser = dist_parser(batch_size=100)
    parser.add_argument('--load-pre-weights', action='store_true')
    parser.add_argument('--epochs', type=int, default=120)
    parser.add_argument('--save-iters', type=int)
    parser.add_argument('--data-switch', action='store_true')
    parser.add_argument('--post-epochs', type=int, default=100)
    parser.add_argument('--eval-epochs', type=int, default=1)
    parser.add_argument('--disable-eval', action='store_true')
    parser.add_argument('--reset', action='store_true')
    args = parser.parse_args()
    dist_check_args(args)
    return args


if __name__ == '__main__':
    main()
