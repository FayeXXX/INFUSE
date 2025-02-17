#!/usr/bin/env bash

splitid=(0 1 2 3 4)
gpuid=$1

for id in "${splitid[@]}" # split id
do
    python osr_cocoop.py  --dataset cifar-10-100 --out_num 50 --loss Softmax --use_default_parameters False --num_workers 32 --split_idx ${id} \
    --batch_size 100 --LR 0.003 --MAX_EPOCH 60 --transform cocoop --backbone ViT-B-32 --gpu ${gpuid} --method cocoop_cat --prec fp16

done


