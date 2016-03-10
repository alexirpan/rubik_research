#!/bin/bash
declare -a arr=("lstm")

for epslen in `seq 5 9`
do
    declare -a name="models_more_data/${epslen}_lstm"
    mkdir $name
    th train_rubiks.lua --epslen $epslen --savedir $name --type $typ --ntrain 100000 --ntest 20000 --gpu 1
done

: '
declare -a name="models_26_lstm"
mkdir $name
th train_rubiks.lua --epsLen 26 --saveDir $name --type lstm
'

