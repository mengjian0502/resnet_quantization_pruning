H#!/usr/bin/env sh

PYTHON="/home/li/.conda/envs/pytorch/bin/python"
imagenet_path="/opt/imagenet/imagenet_compressed"
pretrained_model=./save/resnet18/resnet18_00005_07/checkpoint.pth.tar
############ directory to save result #############
DATE=`date +%Y-%m-%d`

if [ ! -d "$DIRECTORY" ]; then
    mkdir ./save
    mkdir ./save/${DATE}/
fi

############ Configurations ###############
model=resnet18b_w4_a4_tex1
dataset=imagenet
epochs=110
batch_size=256
optimizer=SGD
# add more labels as additional info into the saving path
label_info=

CUDA_VISIBILE_DEVICES=0,2,3 $PYTHON -W ignore main.py --dataset ${dataset} \
    --data_path ${imagenet_path}   \
    --arch ${model} --save_path ./save/resnet18/sgd/test \
    --epochs ${epochs} --learning_rate 0.1 \
    --optimizer ${optimizer} \
    --schedule 30 60 85 95  --gammas 0.1 0.1 0.1 0.1 \
    --batch_size ${batch_size} --workers 16 --ngpu 3 \
    --print_freq 1000  --decay 0.0001 \
    --lamda 0.00025   --ratio 0.7  \
    --clp \
    --a_lambda 0.01 \
    # --swp \
    # --fine_tune \
    # --resume ${pretrained_model} \
    #--evaluate
    #--resume ${pretrained_model} \
    #--ssl \
    #--resume ${pretrained_model} \
    #--fine_tune\
    #--model_only 
    #--evaluate\

