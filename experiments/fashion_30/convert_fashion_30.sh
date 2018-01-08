#!/bin/bash
set -x
mkdir -p data/fashion
files="train-images-idx3-ubyte.gz train-labels-idx1-ubyte.gz t10k-images-idx3-ubyte.gz t10k-labels-idx1-ubyte.gz"

for name in $files ;
do
  if [ ! -f "data/fashion/$name" ] ;
  then
    wget https://github.com/zalandoresearch/fashion-mnist/blob/5e60fd2944e1f66c76303eebaebd6167d1ae1231/data/fashion/$name?raw=true -O data/fashion/$name
  fi
done
for name in $files ;
do
  if [ ! -f "data/fashion/$name" ] ;
  then
    echo error for $name
    exit 1
  fi
done

for seed in 10 20 30 40 50 ;
do
  python -m cfl.scripts.convert_mnist \
      --output-dir parsed_data/fashion_30/$seed \
      --labeled-percent 0.30 \
      --all-random \
      --all-digits \
      --data-path data/fashion \
      --resample \
      --seed $seed
done
