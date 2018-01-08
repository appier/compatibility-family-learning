#!/bin/bash
set -x

python -m cfl.scripts.convert_dyadic_latent \
    --input-dir data/dyadic \
    --output-dir parsed_data/dyadic_latent \
    --latent-dir data/dyadic/latents \
    --meta-path data/monomer/productMeta_simple.txt \
    --sample-val-rate 0.00499800249
