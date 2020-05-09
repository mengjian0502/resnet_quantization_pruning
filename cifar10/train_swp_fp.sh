#!/usr/bin/env sh

PYTHON="/home/mengjian/anaconda3/bin/python3"

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
group_ch=16

# add more labels as additional info into the saving path
label_info=

pretrained_model="./save/resnet20/decay0.0002_fullprecision_multiplecheckpoints_fflf/model_best.pth.tar"

$PYTHON -W ignore main.py --dataset ${dataset} \
    --data_path ./dataset/   \
    --arch ${model} --save_path ./save/vgg7/quant_scheme/sawb_2bit/decay0.0002_w2_a2_sawb_2bit_clamp_fps \
    --epochs ${epochs}  --learning_rate  0.1 \
    --optimizer ${optimizer} \
    --schedule 60 120   --gammas 0.1 0.1\
    --batch_size ${batch_size} --workers 4 --ngpu 1 --gpu_id 2 \
    --print_freq 100 --decay 0.0002 \
    # --lamda 0.0007   --ratio 0.7 \
    # --clp \
    # --a_lambda 0.01 \
    # --resume ${pretrained_model} \
    # --fine_tune \
    # --swp 
    # --w_clp \
    # --b_lambda 0.01 \
