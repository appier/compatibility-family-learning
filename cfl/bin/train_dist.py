import gc
import logging
import os
import shutil
import threading

import tensorflow as tf

from tqdm import trange

from ..input_data import load_data_sets
from ..models.dist import construct_model
from ..ops import normalizer
from ..ops import unnormalizer
from ..utils import IncrementalAverage
from ..utils import dist_eval
from ..utils import load_best_stats
from ..utils import load_model
from ..utils import log_args
from ..utils import monomer_parser
from ..utils import reduce_product

logger = logging.getLogger(__name__)


def enqueue(coord, sess, aux, batch_size):
    (enqueue_op, queues, data) = aux

    while not coord.should_stop():
        batch = data.next_batch(batch_size)
        assert len(queues) == len(batch), '{} != {}'.format(
            len(queues), len(batch))
        feed_dict = {queue: subbatch for queue, subbatch in zip(queues, batch)}
        sess.run(enqueue_op, feed_dict=feed_dict)


def train_loop(sess, model, data, aux, batch_size, start_epoch, epochs,
               log_dir, checkpoint_dir, saver):
    best_saver = tf.train.Saver()
    writer = tf.summary.FileWriter(log_dir, sess.graph)

    nb_train = max(data.train.num_examples_labeled_pos,
                   data.train.num_examples_labeled_neg)
    logger.warning('%d examples', nb_train)
    logger.warning('model: %s', model.get_name())

    nb_batch = nb_train // batch_size

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord, sess=sess)

    # enqueue train
    enqueue_thread = threading.Thread(
        target=enqueue, args=[coord, sess, aux[0], batch_size])
    enqueue_thread.daemon = True
    enqueue_thread.start()

    # enqueue val
    enqueue_thread = threading.Thread(
        target=enqueue, args=[coord, sess, aux[1], batch_size])
    enqueue_thread.daemon = True
    enqueue_thread.start()

    best_dir = os.path.join(checkpoint_dir, 'best_acc_model')
    os.makedirs(best_dir, exist_ok=True)
    best_accuracy_path = os.path.join(best_dir, 'best_accuracy')
    stats = load_best_stats(best_accuracy_path)

    try:
        for e in range(start_epoch, epochs):
            t = trange(nb_batch)

            t.set_description('epoch {}'.format(e))
            train_avg = IncrementalAverage()
            val_avg = IncrementalAverage()

            for i in t:
                summary = model.summary if i != 0 else model.all_summary
                optims = [model.s_optim]

                (summary_str, _, train_acc, val_acc) = sess.run(
                    [summary, optims, model.s_accuracy, model.val_s_accuracy])
                train_avg.add(train_acc)
                val_avg.add(val_acc)
                t.set_postfix(
                    train_acc=train_avg.average, val_acc=val_avg.average)
                writer.add_summary(summary_str, nb_batch * e + i)

            saver.save(
                sess, os.path.join(checkpoint_dir, 'model'), global_step=e)
            gc.collect()

            val_stats = dist_eval(sess, model, batch_size, data.val)
            if val_stats.accuracy > stats.best_accuracy:
                test_stats = dist_eval(sess, model, batch_size, data.test)
                logger.warning(('epoch %d:'
                                ' current error ='
                                ' train: %f val: %f test: %f'
                                ' / auc = val: %f test: %f'), e,
                               1. - train_avg.average, 1. - val_stats.accuracy,
                               1. - test_stats.accuracy, val_stats.auc,
                               test_stats.auc)

                stats.best_accuracy = val_stats.accuracy
                stats.best_auc = val_stats.auc
                stats.best_epoch = e

                best_saver.save(
                    sess,
                    os.path.join(best_dir, 'model'),
                    global_step=stats.best_epoch)

                with open(best_accuracy_path, 'w') as outfile:
                    outfile.write('{}\t{}\t{}'.format(
                        stats.best_epoch, stats.best_accuracy, stats.best_auc))
            else:
                logger.warning(('epoch %d:'
                                ' avg error ='
                                ' train: %f val: %f'), e,
                               1. - train_avg.average, 1. - val_avg.average)

    finally:
        coord.request_stop()
        coord.join(threads)


def train_monomer(data_name, data_root, checkpoint_root, log_root, run_tag,
                  seed, normalize_value, input_shape, batch_size,
                  num_components, latent_size, lr, beta1, beta2, epochs,
                  reg_const, reset):
    tf.set_random_seed(seed)

    input_shape = tuple(input_shape)
    input_size = reduce_product(input_shape)

    data_dir = os.path.join(data_root, data_name)
    checkpoint_dir = os.path.join(checkpoint_root, data_name)
    log_dir = os.path.join(log_root, data_name)

    # load data sets
    data = load_data_sets(data_dir, input_size, seed=seed)

    data_normalizer = normalizer(normalize_value, 0., None, None)
    data_unnormalizer = unnormalizer(normalize_value, 0.)

    model, aux = construct_model(
        input_shape=input_shape,
        latent_size=latent_size,
        normalize_value=normalize_value,
        lr=lr,
        beta1=beta1,
        beta2=beta2,
        num_components=num_components,
        batch_size=batch_size,
        data=data,
        run_tag=run_tag,
        reg_const=reg_const,
        data_normalizer=data_normalizer,
        data_unnormalizer=data_unnormalizer)

    # get paths
    checkpoint_dir = os.path.join(checkpoint_dir, model.get_name())
    log_dir = os.path.join(log_dir, model.get_name())
    if reset:
        for path in [checkpoint_dir, log_dir]:
            if os.path.exists(path):
                shutil.rmtree(path)
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

    logger.info('start session')
    with tf.Session() as sess:
        saver, start_epoch = load_model(sess, checkpoint_dir)
        train_loop(
            sess=sess,
            model=model,
            aux=aux,
            data=data,
            batch_size=batch_size,
            start_epoch=start_epoch,
            epochs=epochs,
            log_dir=log_dir,
            checkpoint_dir=checkpoint_dir,
            saver=saver)


def main():
    args = parse_args()
    log_args(args)
    train_monomer(**vars(args))


def parse_args():
    parser = monomer_parser()
    parser.add_argument('--epochs', type=int, default=120)
    parser.add_argument('--reset', action='store_true')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    main()
