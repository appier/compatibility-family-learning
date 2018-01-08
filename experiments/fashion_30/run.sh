#!/bin/bash
set -x
total_ls=60
for seed in 10 20 30 40 50;
do
  data_name=fashion_30/$seed
  for dist_type in pcd monomer ;
  do
    for nc in 2 3 4 5 ;
    do
      latent_size=$((total_ls/(nc+1)))
      python -m cfl.bin.train \
        --data-name $data_name \
        --model-type conv \
        --data-type sigmoid \
        --dist-type $dist_type \
        --use-threshold \
        --reg-const 5e-4 \
        --num-components $nc \
        --latent-size $latent_size \
        --epochs 50

      python -m cfl.bin.predict \
        --data-name $data_name \
        --model-type conv \
        --data-type sigmoid \
        --dist-type $dist_type \
        --use-threshold \
        --reg-const 5e-4 \
        --num-components $nc \
        --latent-size $latent_size
    done
  done
  for dist_type in siamese ;
  do
    python -m cfl.bin.train \
      --data-name $data_name \
      --model-type conv \
      --data-type sigmoid \
      --dist-type $dist_type \
      --use-threshold \
      --reg-const 5e-4 \
      --num-components 1 \
      --latent-size $total_ls \
      --epochs 50

    python -m cfl.bin.predict \
      --data-name $data_name \
      --model-type conv \
      --data-type sigmoid \
      --dist-type $dist_type \
      --use-threshold \
      --reg-const 5e-4 \
      --num-components 1 \
      --latent-size $total_ls
  done
done
