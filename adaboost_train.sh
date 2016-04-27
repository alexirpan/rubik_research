#!/bin/bash
# Only 20 epochs because of time issues
declare -a epslen=9
declare -a prob=0
declare -a name="models_ada_mislabel/${epslen}_lstm_prob${prob}"
mkdir $name
th train_rubiks.lua --type lstm --epslen $epslen --savedir $name --epochs 20 --ntrain 50000 --adaboost 1 --batchsize 1 --keepprob $prob --gpu 1 | tee "${name}/stdout"

