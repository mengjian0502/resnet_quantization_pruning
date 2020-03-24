H#!/usr/bin/env sh

PYTHON="/home/lyang/anaconda3/bin/python3"
imagenet_path="~/dataset/imagenet"
pretrained_model=./save/resnet18/resnet18_00005_07/checkpoint.pth.tar
############ directory to save result #############
DATE=`date +%Y-%m-%d`

if [ ! -d "$DIRECTORY" ]; then
    mkdir ./save
    mkdir ./save/${DATE}/
fi

############ Configurations ###############
model=resnet18b_ff_lf_w4_a4_tex1
dataset=imagenet
epochs=60
batch_size=256
optimizer=SGD
# add more labels as additional info into the saving path
label_info=

$PYTHON main.py --dataset ${dataset} \
    --data_path ${imagenet_path}   \
    --arch ${model} --save_path ./save/resnet18/sgd/resnet18_baseline \
    --epochs ${epochs} --learning_rate 0.1 \
    --optimizer ${optimizer} \
    --schedule 30 40 45  --gammas 0.1 0.1 0.1 \
    --batch_size ${batch_size} --workers 16 --ngpu 4 --gpu_id 0  \
    --print_freq 1000  --decay 0.0001 \
    --lamda 0.0005   --ratio 0.7  \
    # --swp \
    # --resume ${pretrained_model} \
    #--evaluate
    #--resume ${pretrained_model} \
    #--fine_tune\
    #--ssl \
    #--resume ${pretrained_model} \
    #--fine_tune\
    #--model_only 
    #--evaluate\

