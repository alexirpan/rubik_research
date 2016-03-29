#!/bin/bash
declare -a epslen=26
declare -a name="models_streaming/${epslen}_lstm"
mkdir $name
th train_rubiks.lua --epslen $epslen --savedir $name --model models_streaming/26_lstm/rubiks_epoch40 --epochs 40 --ntrain 50000 --ntest 10000 --gpu 1

