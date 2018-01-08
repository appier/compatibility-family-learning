#!/bin/bash
set -x
nc=3
latent_size=64
data_name=dyadic_latent
dist_types=${@:-pcd monomer siamese}
for dist_type in $dist_types ;
do
    if [ $dist_type == "pcd" ] || [ $dist_type == "monomer" ] ;
    then
        python -m cfl.bin.train \
            --data-name $data_name \
            --model-type linear \
            --data-type linear \
            --data-norm 31.9098 \
            --data-switch \
            --input-shape 1024 \
            --dist-type $dist_type \
            --pos-weight 0.0625 \
            --use-threshold \
            --num-components $nc \
            --latent-size $latent_size \
            --epochs 5

        python -m cfl.bin.predict \
            --data-name $data_name \
            --model-type linear \
            --data-type linear \
            --data-norm 31.9098 \
            --input-shape 1024 \
            --dist-type $dist_type \
            --pos-weight 0.0625 \
            --use-threshold \
            --num-components $nc \
            --latent-size $latent_size
    fi
    if [ $dist_type == "siamese" ] ;
    then
        python -m cfl.bin.train \
            --data-name $data_name \
            --model-type linear \
            --data-type linear \
            --data-norm 31.9098 \
            --input-shape 1024 \
            --dist-type $dist_type \
            --caffe-margin 100. \
            --pos-weight 0.0625 \
            --num-components 1 \
            --latent-size 256 \
            --epochs 5

        python -m cfl.bin.predict \
            --data-name $data_name \
            --model-type linear \
            --data-type linear \
            --data-norm 31.9098 \
            --input-shape 1024 \
            --dist-type $dist_type \
            --caffe-margin 100. \
            --pos-weight 0.0625 \
            --num-components 1 \
            --latent-size 256
    fi
done
