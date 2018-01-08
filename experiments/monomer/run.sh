#!/bin/bash
set -x
seed=0
nc=4
names=${@:-Baby Boys Girls Men Women}

for name in $names ;
do
  for rel in also_viewed ;
  do
    for latent_size in 10 ;
    do
      python -m cfl.bin.train_dist \
        --data-name monomer/$name-$rel \
        --input-shape 4096 \
        --num-components $nc \
        --latent-size $latent_size \
        --normalize-value 58.388599 \
        --seed $seed \
        --epochs 200

      python -m cfl.bin.predict_dist \
        --data-name monomer/$name-$rel \
        --input-shape 4096 \
        --num-components $nc \
        --latent-size $latent_size \
        --normalize-value 58.388599 \
        --seed $seed
    done
  done

  for rel in also_bought ;
  do
    for latent_size in 20 ;
    do
      python -m cfl.bin.train_dist \
        --data-name monomer/$name-$rel \
        --input-shape 4096 \
        --num-components $nc \
        --latent-size $latent_size \
        --normalize-value 58.388599 \
        --seed $seed \
        --epochs 200

      python -m cfl.bin.predict_dist \
        --data-name monomer/$name-$rel \
        --input-shape 4096 \
        --num-components $nc \
        --latent-size $latent_size \
        --normalize-value 58.388599 \
        --seed $seed
    done
  done
done
