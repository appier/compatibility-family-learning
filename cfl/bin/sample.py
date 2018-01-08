import logging
import os

import tensorflow as tf

from ..input_data import load_data_sets
from ..models.cfl import construct_model
from ..ops import dist_ae_transformer
from ..ops import dist_normalizer
from ..ops import dist_transformer
from ..utils import dist_check_args
from ..utils import dist_parser
from ..utils import dist_sample
from ..utils import dist_sample_near
from ..utils import dist_sample_project_disc
from ..utils import load_model, reduce_product

logger = logging.getLogger(__name__)


def main(data_name, data_root, sample_root, checkpoint_root, log_root, run_tag,
         seed, source_shape, input_shape, ae_shape, batch_size, data_scale,
         pos_weight, data_mean, data_norm, data_type, data_mirror,
         data_random_crop, data_is_image, data_is_double, data_disable_double,
         latent_norm, latent_shape, model_type, gan_type, num_components,
         latent_size, raw_latent, caffe_margin, gan, cgan, t_dim, dist_type,
         act_type, use_threshold, lr, beta1, beta2, z_dim, z_stddev, g_dim,
         g_lr, g_beta1, g_beta2, m_prj, m_enc, d_dim, d_lr, d_beta1, d_beta2,
         lambda_dra, lambda_gp, lambda_m, directed, data_directed, reg_const,
         reorder, sample_type):
    tf.set_random_seed(seed)

    input_size = reduce_product(input_shape)
    source_size = reduce_product(source_shape) if source_shape else input_size

    data_dir = os.path.join(data_root, data_name)
    sample_dir = os.path.join(sample_root, data_name)
    checkpoint_dir = os.path.join(checkpoint_root, data_name)

    # load data sets
    data = load_data_sets(
        data_dir,
        source_size,
        is_image=data_is_image,
        raw_latent=raw_latent,
        is_double=data_is_double,
        directed=directed or data_directed,
        reorder=reorder,
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

    model, _ = construct_model(
        source_shape=source_shape,
        is_double=data_is_double,
        disable_double=data_disable_double,
        latent_shape=latent_shape,
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
        enable_input_producer=False)

    # get paths
    checkpoint_dir = os.path.join(checkpoint_dir, model.get_name())
    sample_dir = os.path.join(sample_dir, model.get_name(), sample_type)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    logger.info('start session')
    with tf.Session(config=config) as sess:
        saver, start_epoch = load_model(sess, os.path.join(checkpoint_dir))

        if sample_type == 'project':
            dist_sample(
                sess=sess,
                model=model,
                data=data.test,
                batch_size=batch_size,
                sample_dir=sample_dir,
                output_name='test')
        elif sample_type == 'near':
            dist_sample_near(
                sess=sess,
                model=model,
                data=data.test,
                batch_size=batch_size,
                sample_dir=sample_dir,
                output_name='test')
        elif sample_type == 'project_disc':
            dist_sample_project_disc(
                sess=sess,
                model=model,
                data=data.test,
                batch_size=batch_size,
                sample_dir=sample_dir,
                output_name='test')


def start():
    log_format = '%(asctime)s [%(levelname)-5.5s] [%(name)s]  %(message)s'
    logging.basicConfig(format=log_format, level=logging.WARNING)
    args = parse_args()
    main(**vars(args))


def parse_args():
    parser = dist_parser(batch_size=50)
    parser.add_argument('--sample-root', default='samples')
    parser.add_argument('--reorder', action='store_true')
    parser.add_argument(
        '--sample-type',
        choices=('project', 'near', 'project_disc'),
        default='project')
    args = parser.parse_args()
    dist_check_args(args)
    return args


if __name__ == '__main__':
    start()
