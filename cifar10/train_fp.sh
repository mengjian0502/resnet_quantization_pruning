#!/usr/bin/env sh

PYTHON="/home/mengjian/anaconda3/envs/neurosim_test/bin/python3"

############ directory to save result #############
DATE=`date +%Y-%m-%d`

if [ ! -d "$DIRECTORY" ]; then
    mkdir ./save
    mkdir ./dataset
    mkdir ./save/${DATE}/
fi

############ Configurations ###############
model=vgg7
dataset=cifar10
epochs=200
batch_size=128
optimizer=SGD
wd=0.0005
# a_lambda=0.01

eval_model="./save/resnet20/full_precsion/decay0.0002_w32_a32_fullprecision/model_best.pth.tar"

$PYTHON -W ignore main.py --dataset ${dataset} \
    --data_path ./dataset/   \
    --arch ${model} --save_path ./save/resnet20/full_precsion/decay${wd}_w32_a32_fullprecision_eval \
    --epochs ${epochs}  --learning_rate  0.1 \
    --optimizer ${optimizer} \
    --schedule 60 120   --gammas 0.1 0.1 \
    --batch_size ${batch_size} --workers 4 --ngpu 1 --gpu_id 2 \
    --print_freq 100 --decay ${wd} \
    --resume ${eval_model} \
    --evaluate