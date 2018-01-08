#!/bin/bash
data_name=${@:-fashion_30}
total_ls=60
seeds="10 20 30 40 50"
for dist_type in siamese ;
do
  filenames=
  datapaths=
  for seed in $seeds ;
  do
    filenames="$filenames predicts/$data_name/$seed/cfl_${dist_type}_conv_sigmoid_ls_${total_ls}_ut_reg_0.0005/"
    datapaths="$datapaths parsed_data/$data_name/$seed"
  done
  python -m cfl.bin.evaluate_total \
    --avg \
    --data-path $datapaths \
    --predict-paths $filenames \
    --name $data_name-$dist_type-sig
done

for dist_type in monomer pcd ;
do
  for nc in 2 3 4 5 ;
  do
    latent_size=$((total_ls/(nc+1)))
    filenames=
    datapaths=
    for seed in $seeds ;
    do
      filenames="$filenames predicts/$data_name/$seed/cfl_${dist_type}_conv_sigmoid_ls_${latent_size}_nc_${nc}_ut_reg_0.0005/"
      datapaths="$datapaths parsed_data/$data_name/$seed"
    done
    python -m cfl.bin.evaluate_total \
      --avg \
      --data-path $datapaths \
      --predict-paths $filenames \
      --name $data_name-$dist_type-sig-$nc
  done
done
