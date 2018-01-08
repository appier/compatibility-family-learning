#!/bin/bash
set -x
run_data=${@:-bottom_to_other top_to_other shoe_to_other}
for data in $run_data ;
do
    data_name=polyvore_random/$data
    for dist_type in pcd monomer ;
    do
      python -m cfl.bin.train \
        --data-name $data_name \
        --model-type linear \
        --data-type tanh \
        --data-mean 0.5 \
        --data-norm 0.5 \
        --data-is-image \
        --data-is-double \
        --latent-shape 2048 \
        --input-shape 64 64 3 \
        --dist-type $dist_type \
        --pos-weight 0.25 \
        --use-threshold \
        --num-components 4 \
        --latent-size 20 \
        --epochs 10

      python -m cfl.bin.predict \
        --data-name $data_name \
        --model-type linear \
        --data-type tanh \
        --data-mean 0.5 \
        --data-norm 0.5 \
        --data-is-image \
        --data-is-double \
        --latent-shape 2048 \
        --input-shape 64 64 3 \
        --dist-type $dist_type \
        --pos-weight 0.25 \
        --use-threshold \
        --num-components 4 \
        --latent-size 20
    done

    for dist_type in siamese ;
    do
      python -m cfl.bin.train \
        --data-name $data_name \
        --model-type linear \
        --data-type tanh \
        --data-mean 0.5 \
        --data-norm 0.5 \
        --data-is-image \
        --data-is-double \
        --latent-shape 2048 \
        --input-shape 64 64 3 \
        --dist-type $dist_type \
        --pos-weight 0.25 \
        --use-threshold \
        --num-components 1 \
        --latent-size 100 \
        --epochs 10

      python -m cfl.bin.predict \
        --data-name $data_name \
        --model-type linear \
        --data-type tanh \
        --data-mean 0.5 \
        --data-norm 0.5 \
        --data-is-image \
        --data-is-double \
        --latent-shape 2048 \
        --input-shape 64 64 3 \
        --dist-type $dist_type \
        --pos-weight 0.25 \
        --use-threshold \
        --num-components 1 \
        --latent-size 100
    done
done
