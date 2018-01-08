#!/bin/bash
set -x
python -m cfl.scripts.convert_mnist --output-dir parsed_data/mnist_30 --labeled-percent 0.30 --all-random --all-digits
