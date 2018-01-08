#!/bin/bash
set -x
run_data=top_to_other
epochs=10
nc=2
latent_size=20

for data in $run_data ;
do
    data_name=polyvore_random/$data
    python -m cfl.bin.train \
        --data-name $data_name \
        --model-type linear \
        --data-type tanh \
        --data-mean 0.5 \
        --data-norm 0.5 \
        --data-directed \
        --data-is-image \
        --data-is-double \
        --latent-shape 2048 \
        --input-shape 64 64 3 \
        --dist-type pcd \
        --lambda-m 0.5 \
        --use-threshold \
        --num-components $nc \
        --latent-size $latent_size \
        --epochs $epochs

    python -m cfl.bin.train \
        --load-pre-weights \
        --data-name $data_name \
        --model-type linear \
        --data-type tanh \
        --data-mean 0.5 \
        --data-norm 0.5 \
        --data-directed \
        --data-is-image \
        --data-is-double \
        --latent-shape 2048 \
        --input-shape 64 64 3 \
        --dist-type pcd \
        --lambda-m 0.5 \
        --use-threshold \
        --num-components $nc \
        --latent-size $latent_size \
        --m-prj 0.3 \
        --m-enc 0.05 \
        --d-lr 0.0002 \
        --d-beta1 0.5 \
        --g-lr 0.0002 \
        --g-beta1 0.5 \
        --gan \
        --gan-type srgan \
        --lambda-gp 0.5 \
        --epochs $epochs \
        --post-epochs 400
done
