#!/usr/bin/env sh

PYTHON="/home/mengjian/anaconda3/envs/myenv_pc/bin/python3"

############ directory to save result #############
DATE=`date +%Y-%m-%d`

if [ ! -d "$DIRECTORY" ]; then
    mkdir ./save
    mkdir ./dataset
    mkdir ./save/${DATE}/
fi

############ Configurations ###############
model=preresnet20
dataset=cifar10
epochs=200
batch_size=128
optimizer=SGD
group_ch=16

# add more labels as additional info into the saving path
label_info=

$PYTHON -W ignore main.py --dataset ${dataset} \
    --data_path ./dataset/   \
    --arch ${model} --save_path ./save/preresnet20/exp022720/decay0.0005_baseline \
    --epochs ${epochs}  --learning_rate  0.1 \
    --optimizer ${optimizer} \
    --schedule 60 120   --gammas 0.1 0.1\
    --batch_size ${batch_size} --workers 4 --ngpu 1 --gpu_id 2 \
    --print_freq 100 --decay 0.0002 \
    --lamda 0.007   --ratio 0.7 \
    --clp \
    --a_lambda 0.01 \
    # --resume ${pretrained_model} \
    # --fine_tune \
    # --swp 
