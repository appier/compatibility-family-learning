#!/bin/bash
set -x
data_name=mnist_30

# check different lambda_m
for lm in 0.0 0.1 0.5 1.0 2.0 5.0 ; do
    python -m cfl.bin.train \
      --data-name $data_name \
      --model-type conv \
      --data-type sigmoid \
      --dist-type pcd \
      --lambda-m $lm \
      --use-threshold \
      --num-components 2 \
      --latent-size 20 \
      --epochs 50

    python -m cfl.bin.train \
      --load-pre-weights \
      --data-name $data_name \
      --model-type conv \
      --data-type sigmoid \
      --dist-type pcd \
      --lambda-m $lm \
      --use-threshold \
      --m-prj 0.5 \
      --m-enc 0.1 \
      --d-lr 0.001 \
      --d-beta1 0.9 \
      --g-lr 0.001 \
      --g-beta1 0.9 \
      --num-components 2 \
      --latent-size 20 \
      --gan \
      --epochs 50 \
      --post-epochs 400
done

# check different m_enc
for m_enc in 0.0 0.1 0.5 1.0 2.0 5.0 ; do
    python -m cfl.bin.train \
      --load-pre-weights \
      --data-name $data_name \
      --model-type conv \
      --data-type sigmoid \
      --dist-type pcd \
      --lambda-m 0.5 \
      --use-threshold \
      --m-prj 0.5 \
      --m-enc $m_enc \
      --d-lr 0.001 \
      --d-beta1 0.9 \
      --g-lr 0.001 \
      --g-beta1 0.9 \
      --num-components 2 \
      --latent-size 20 \
      --gan \
      --epochs 50 \
      --post-epochs 400
done

# check different m_prj
for m_prj in 0.0 0.1 0.5 1.0 2.0 5.0 ; do
    python -m cfl.bin.train \
      --load-pre-weights \
      --data-name $data_name \
      --model-type conv \
      --data-type sigmoid \
      --dist-type pcd \
      --lambda-m 0.5 \
      --use-threshold \
      --m-prj $m_prj \
      --m-enc 0.1 \
      --d-lr 0.001 \
      --d-beta1 0.9 \
      --g-lr 0.001 \
      --g-beta1 0.9 \
      --num-components 2 \
      --latent-size 20 \
      --gan \
      --epochs 50 \
      --post-epochs 400
done

./experiments/$data_name/sample_gen_params.sh
