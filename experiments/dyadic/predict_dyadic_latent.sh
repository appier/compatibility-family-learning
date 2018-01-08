#!/bin/bash
set -x

for split in train val test ;
do
    python -m cfl.scripts.predict_dyadic \
        --model-file experiments/dyadic/latent_googlenet-siamese.prototxt \
        --weight-file models/googlenet-siamese-final.caffemodel \
        --id-path data/dyadic/${split}_ids.parsed.txt \
        --input-dir data/dyadic/images \
        --output-dir data/dyadic/latents
done
