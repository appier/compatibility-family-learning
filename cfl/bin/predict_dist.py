import logging
import os

import tensorflow as tf

from ..input_data import load_data_sets
from ..models.dist import construct_model
from ..ops import normalizer
from ..ops import unnormalizer
from ..utils import dist_predict
from ..utils import load_model, reduce_product
from ..utils import monomer_parser

logger = logging.getLogger(__name__)


def predict_monomer(data_name, data_root, checkpoint_root, log_root,
                    predict_root, run_tag, seed, normalize_value, input_shape,
                    batch_size, num_components, latent_size, lr, beta1, beta2,
                    reg_const):
    tf.set_random_seed(seed)

    input_shape = tuple(input_shape)
    input_size = reduce_product(input_shape)

    data_dir = os.path.join(data_root, data_name)
    checkpoint_dir = os.path.join(checkpoint_root, data_name)
    predict_dir = os.path.join(predict_root, data_name)

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
    predict_dir = os.path.join(predict_dir, model.get_name())

    logger.info('start session')
    with tf.Session() as sess:
        saver, start_epoch = load_model(sess,
                                        os.path.join(checkpoint_dir,
                                                     'best_acc_model'))
        dist_predict(
            sess=sess,
            model=model,
            data=data.train,
            batch_size=batch_size,
            predict_dir=predict_dir,
            output_name='predict_train_acc.txt')

        dist_predict(
            sess=sess,
            model=model,
            data=data.val,
            batch_size=batch_size,
            predict_dir=predict_dir,
            output_name='predict_val_acc.txt')

        dist_predict(
            sess=sess,
            model=model,
            data=data.test,
            batch_size=batch_size,
            predict_dir=predict_dir,
            output_name='predict_acc.txt')


def main():
    log_format = '%(asctime)s [%(levelname)-5.5s] [%(name)s]  %(message)s'
    logging.basicConfig(format=log_format, level=logging.WARNING)
    args = parse_args()
    predict_monomer(**vars(args))


def parse_args():
    parser = monomer_parser()
    parser.add_argument('--predict-root', default='predicts')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    main()
