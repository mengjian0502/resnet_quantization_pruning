#!/usr/bin/env sh

PYTHON="/home/li/.conda/envs/pytorch/bin/python"
imagenet_path="/opt/imagenet/imagenet_compressed/"
pretrained_model=./save/new_resnet18/hc_naive_5e5/checkpoint.pth.tar
############ directory to save result #############
DATE=`date +%Y-%m-%d`

if [ ! -d "$DIRECTORY" ]; then
    mkdir ./save
    mkdir ./save/${DATE}/
fi

############ Configurations ###############
model=resnet18b_ff_lf_tex1
dataset=imagenet
epochs=100
batch_size=256
optimizer=Adam
# add more labels as additional info into the saving path
label_info=

$PYTHON main.py --dataset ${dataset} \
    --data_path ${imagenet_path}   \
    --arch ${model} --save_path ./save/new_resnet18/hc_5e4_naivecombine \
    --epochs ${epochs} --learning_rate 0.0001 \
    --optimizer ${optimizer} \
    --schedule 30 40 45  --gammas 0.2 0.2 0.5 \
    --batch_size ${batch_size} --workers 16 --ngpu 4 --gpu_id 0  \
    --print_freq 500  --decay 0.000005 \
    --lamda 0.0005   --ratio 0.7 \
    --swp \
    # --resume ${pretrained_model} \
    # --evaluate \
    #--model_only 
    #--evaluate\

