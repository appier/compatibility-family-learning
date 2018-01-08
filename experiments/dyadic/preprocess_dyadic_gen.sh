#!/bin/bash
set -x
python -m cfl.scripts.convert_dyadic_generation \
    --input-dir parsed_data/dyadic_latent \
    --image-dir data/dyadic/images \
    --output-dir parsed_data/dyadic_gen_all \
    --preserve
