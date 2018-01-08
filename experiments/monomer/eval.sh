#!/bin/bash
seed=0
nc=4
names="Men Women Boys Girls Baby"

for name in $names ;
do
  for rel in also_bought ;
  do
    data_name=monomer/$name-$rel
    for latent_size in 20 ;
    do
      python -m cfl.bin.evaluate_total \
        --data-path parsed_data/$data_name \
        --predict-paths \
        predicts/$data_name/linear_dist_ls_${latent_size}_nc_${nc}_reg_0.0_norm_58.388599/ \
        --name pcd-$name-$rel
    done
  done
  for rel in also_viewed ;
  do
    data_name=monomer/$name-$rel
    for latent_size in 10 ;
    do
      python -m cfl.bin.evaluate_total \
        --data-path parsed_data/$data_name \
        --predict-paths \
        predicts/$data_name/linear_dist_ls_${latent_size}_nc_${nc}_reg_0.0_norm_58.388599/ \
        --name pcd-$name-$rel
    done
  done
done
