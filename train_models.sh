#!/bin/bash
declare -a epslen=26
declare -a name="models_streaming/${epslen}_lstm"
mkdir $name
th train_rubiks.lua --epslen $epslen --savedir $name --model "${name}/rubiks_epoch80" --epochs 40 --ntrain 50000 --ntest 10000 --gpu 1

declare -a epslen=9
declare -a name="models_streaming/${epslen}_lstm"
mkdir $name
th train_rubiks.lua --epslen $epslen --savedir $name --model "${name}/rubiks_epoch40" --epochs 80 --ntrain 50000 --ntest 10000 --gpu 1

