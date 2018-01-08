#!/bin/bash
set -x
rootdir=Monomer
if [ ! -d "$rootdir/args" ] ;
then
    git clone https://github.com/Taywee/args.git $rootdir/args
fi
if [ ! -f "$rootdir/src/split.cpp" ] ;
then
    patch -d $rootdir -p1 < experiments/monomer/monomer.patch
fi
make -C $rootdir
