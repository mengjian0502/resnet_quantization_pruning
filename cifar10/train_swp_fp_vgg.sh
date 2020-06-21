#!/usr/bin/env sh

# PYTHON="/home/mengjian/anaconda3/bin/python3"
PYTHON="/home/mengjian/anaconda3/envs/neurosim_test/bin/python3"

############ directory to save result #############
DATE=`date +%Y-%m-%d`

if [ ! -d "$DIRECTORY" ]; then
    mkdir ./save
    mkdir ./dataset
    mkdir ./save/${DATE}/
fi

############ Configurations ###############
model=vgg7_quant
dataset=cifar10
epochs=200
batch_size=128
optimizer=SGD
group_ch=16

# add more labels as additional info into the saving path
label_info=

# pretrained_model="./save/vgg7/full_precision/decay0.0001_w32_a32/model_best.pth.tar"
pretrained_model="./save/vgg7_bn_fuse/full_precision/decay0.0001_w32_a4/model_best.pth.tar"

$PYTHON -W ignore main.py --dataset ${dataset} \
    --data_path ./dataset/   \
    --arch ${model} --save_path ./save/vgg7_bn_fuse/full_precision/decay0.0001_w4_a4 \
    --epochs ${epochs}  --learning_rate  0.001 \
    --optimizer ${optimizer} \
    --schedule 60 120   --gammas 0.1 0.1\
    --batch_size ${batch_size} --workers 4 --ngpu 1 --gpu_id 2 \
    --print_freq 100 --decay 0.0001 \
    --clp \
    --a_lambda 0.001 \
    --resume ${pretrained_model} \
    --fine_tune \
    --bn_fuse \
