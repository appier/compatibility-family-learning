#!/bin/bash
set -x
python -m cfl.scripts.resize_images \
    --image-dir data/dyadic/original_images \
    --output-dir data/dyadic/images \
    --input-file data/dyadic/all_id_pairs.txt \
    --output-shape 256 256 3
