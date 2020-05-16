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
model=tern_resnet20
dataset=cifar10
epochs=200
batch_size=128
optimizer=SGD
wd=0.0002
a_lambda=0.0002

pretrained_model="./save/resnet20/decay0.0002_fullprecision_multiplecheckpoints_fflf/model_best.pth.tar"

# add more labels as additional info into the saving path
label_info=

$PYTHON -W ignore main.py --dataset ${dataset} \
    --data_path ./dataset/   \
    --arch ${model} --save_path ./save/resnet20/quant_scheme/sawb_2bit/decay${wd}_alambda${a_lambda}_finetuneTrue_w2_a2_clamp_2bit_4bit_sc_4bitalpha_w \
    --epochs ${epochs}  --learning_rate  0.1 \
    --optimizer ${optimizer} \
    --schedule 60 120   --gammas 0.1 0.1\
    --batch_size ${batch_size} --workers 4 --ngpu 1 --gpu_id 2 \
    --print_freq 100 --decay ${wd} \
    --clp \
    --a_lambda ${a_lambda} \