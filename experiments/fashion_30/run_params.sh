#!/bin/bash
set -x
for total_ls in 15 30 45 60 75 90 105 ;
do
  for seed in 10 20 30 40 50;
  do
    data_name=fashion_30/$seed
    for dist_type in pcd ;
    do
      for nc in 2 ;
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
  done
done
