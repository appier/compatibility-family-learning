#!/bin/bash
set -x
python -m cfl.scripts.convert_json data/amazon/metadata.json.gz data/amazon/metadata.really.json.gz
python -m cfl.scripts.preprocess_dyadic --meta-path data/amazon/metadata.really.json.gz --data-path data/dyadic
