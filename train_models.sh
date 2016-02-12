#!/bin/bash
declare -a arr=("full" "rnn")

for epslen in `seq 2 9`
do
    for typ in "${arr[@]}"
    do
        declare -a name="models_${epslen}_${typ}_fixed"
        mkdir $name
        th train_rubiks.lua --epsLen $epslen --saveDir $name --type $typ
    done
done
