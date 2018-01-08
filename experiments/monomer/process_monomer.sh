#!/bin/bash
set -x
for name in Baby Men Women Boys Girls ;
do
    for rel in also_bought also_viewed ;
    do
        python -m cfl.scripts.split_meta data/amazon/image_features_Clothing_Shoes_and_Jewelry.b data/monomer/productMeta_simple.txt parsed_data/monomer/${name}-${rel}
    done
done
