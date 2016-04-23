#!/bin/bash
declare -a epslen=9
declare -a name="models_ada/${epslen}_lstm"
mkdir $name
th train_rubiks.lua --type lstm --epslen $epslen --savedir $name --epochs 40 --ntrain 50000 --adaboost 1 --batchsize 1 --keepprob 0.9 --gpu 1 | tee "${name}/stdout"
