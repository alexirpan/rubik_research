#!/bin/bash
declare -a arr=("lstm")

: '
for epslen in `seq 2 9`
do
    for typ in "${arr[@]}"
    do
        declare -a name="models_${epslen}_${typ}"
        mkdir $name
        th train_rubiks.lua --epsLen $epslen --saveDir $name --type $typ
    done
done
'

declare -a name="models_26_lstm"
mkdir $name
th train_rubiks.lua --epsLen 26 --saveDir $name --type lstm

