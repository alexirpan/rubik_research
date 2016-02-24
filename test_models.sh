#!/bin/bash
declare -a arr=("full" "rnn" "lstm" "fulltwo")

for epslen in `seq 2 9`
do
    for typ in "${arr[@]}"
    do
        declare -a name="models_${epslen}_${typ}"
        th test_rubiks.lua --model "${name}/rubiks_best" --savefile "test_results/${name}"
    done
done
