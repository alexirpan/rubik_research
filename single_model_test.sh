#!/bin/bash
# Tests a single model on its ability to solve scrambles of many lengths

if [ ${1} == "" ]; then
    echo "Expected a model directory as first argument"
    exit 1
fi

declare -a modeldir=$1
declare -a model="${modeldir}/rubiks_best"
declare -a savedir="${modeldir}_tests"

mkdir $savedir
for epslen in `seq 1 26`
do
    th test_rubiks.lua --model $model --scramblelen $epslen --savefile "${savedir}/${epslen}"
done
