#!/bin/bash
for epslen in {9,26}
do
    declare -a name="models_streaming/${epslen}_lstm"
    mkdir $name
    th train_rubiks.lua --epslen $epslen --savedir $name --type lstm --ntrain 50000 --ntest 10000 --gpu 1
done

