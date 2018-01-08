#!/bin/bash
set -x
python -m cfl.scripts.convert_polyvore \
    --input-dir data/polyvore \
    --output-dir parsed_data/polyvore_random
