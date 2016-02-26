#!/bin/bash
# Tests a single model on its ability to solve scrambles of many lengths

declare -a modeldir="models_26_lstm"
declare -a model="${modeldir}/rubiks_best"
declare -a savedir="${modeldir}_tests"

mkdir $savedir
for epslen in `seq 1 26`
do
    th test_rubiks.lua --model $model --scramblelen $epslen --savefile "${savedir}/${epslen}"
done
