#!/bin/bash
declare -a arr=("full" "rnn")

for epslen in `seq 2 9`
do
    for typ in "${arr[@]}"
    do
        mkdir "models_${epslen}_${typ}"
        th train_rubiks.lua --epsLen $epslen --saveDir "models_${epslen}_${typ}" --type $typ
    done
done
