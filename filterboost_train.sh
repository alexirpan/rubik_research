#!/bin/bash
declare -a epslen=9
declare -a name="models_filter/${epslen}_lstm"
mkdir $name
th train_rubiks.lua --type lstm --epslen $epslen --savedir $name --epochs 40 --ntrain 50000 --ntest 10000 --filboost 1 --nboosttest 5000 --gpu 1 | tee "${name}/stdout"

declare -a epslen=26
declare -a name="models_filter/${epslen}_lstm"
mkdir $name
th train_rubiks.lua --type lstm --epslen $epslen --savedir $name --epochs 40 --ntrain 50000 --ntest 10000 --filboost 1 --nboosttest 5000 --gpu 1 | tee "${name}/stdout"
