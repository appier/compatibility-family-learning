#!/bin/bash
set -x
nc=2
latent_size=64
data_name=dyadic_gen_all
for sample_type in near project_disc ;
do
    python -m cfl.bin.sample \
        --data-name $data_name \
        --model-type linear \
        --data-type tanh \
        --data-mean 0.5 \
        --data-norm 0.5 \
        --data-directed \
        --latent-norm 31.9098 \
        --data-is-image \
        --data-is-double \
        --latent-shape 1024 \
        --input-shape 64 64 3 \
        --dist-type pcd \
        --lambda-m 0.5 \
        --use-threshold \
        --num-components $nc \
        --latent-size $latent_size \
        --m-prj 0.2 \
        --m-enc 0.05 \
        --d-lr 0.0002 \
        --d-beta1 0.5 \
        --g-lr 0.0002 \
        --g-beta1 0.5 \
        --gan \
        --gan-type srgan \
        --lambda-gp 0.5 \
        --sample-type $sample_type
done
