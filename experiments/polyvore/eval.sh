#!/bin/bash
run_data=${@:-bottom_to_other top_to_other shoe_to_other}

for data in $run_data ;
do
    data_name=polyvore_random/$data
    for dist_type in pcd monomer ;
    do
        python -m cfl.bin.evaluate_total \
            --data-path parsed_data/$data_name \
            --predict-paths \
            predicts/$data_name/cfl_${dist_type}_linear_pw_0.25_tanh_ls_20_nc_4_ut_norm_0.5/ \
            --name $data-$dist_type-4-20 \
            --auc-model
    done
    for dist_type in siamese ;
    do
        python -m cfl.bin.evaluate_total \
            --data-path parsed_data/$data_name \
            --predict-paths \
            predicts/$data_name/cfl_${dist_type}_linear_pw_0.25_tanh_ls_100_ut_norm_0.5/ \
            --name $data-$dist_type-100 \
            --auc-model
    done
done
