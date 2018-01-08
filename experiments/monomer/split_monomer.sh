#!/bin/bash
set -x
for name in Baby Men Women Boys Girls ;
do
    for rel in also_bought also_viewed ;
    do
        ./Monomer/split \
            -f data/amazon/image_features_Clothing_Shoes_and_Jewelry.b \
            -d 4096 \
            -m data/monomer/productMeta_simple.txt \
            --cp Clothing\ Shoes\ \&\ Jewelry\|$name \
            --layer 2 \
            --graph_path data/monomer/$rel.txt \
            --dup_path data/monomer/duplicate_list.txt \
            --output_path parsed_data/monomer/${name}-${rel}
    done
done
