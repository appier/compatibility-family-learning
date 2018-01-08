#!/bin/bash
data_name=dyadic_latent
dist_types=${@:-pcd monomer siamese}
nc=3
latent_size=64


for dist_type in $dist_types ;
do
    if [ $dist_type == "pcd" ] || [ $dist_type == "monomer" ] ;
    then
        python -m cfl.bin.evaluate_total \
            --auc-model \
            --data-path parsed_data/$data_name \
            --predict-paths predicts/$data_name/cfl_${dist_type}_linear_pw_0.0625_linear_ls_${latent_size}_nc_${nc}_ut_norm_31.9098 \
            --name $data_name-$dist_type
    fi
    if [ $dist_type == "siamese" ] ;
    then
        python -m cfl.bin.evaluate_total \
            --auc-model \
            --data-path parsed_data/$data_name \
            --predict-paths predicts/$data_name/cfl_${dist_type}_linear_pw_0.0625_margin_100.0_linear_ls_256_norm_31.9098 \
            --name $data_name-$dist_type
    fi
done
