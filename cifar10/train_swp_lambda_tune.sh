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
model=tern_resnet56
dataset=cifar10
epochs=200
batch_size=128
optimizer=SGD
group_ch=16

####### Lambda Tuning #######
ub=0.005
lb=0.0005
diff=0.0005

# add more labels as additional info into the saving path
label_info=

# pretrained_model="./save/resnet56/decay0.0002_fflf/model_best.pth.tar" # full precision model
pretrained_model="./save/resnet20/exp012720/decay0.0002_fullprecision_multiplecheckpoints_fflf/model_best.pth.tar"

for i in $(seq ${lb} ${diff} ${ub})
do
    $PYTHON main.py --dataset ${dataset} \
        --data_path ./dataset/   \
        --arch ${model} --save_path ./save/resnet20/exp012720/sawb_tanh_optimal_alpha_lambda_tuning_resnet20/decay0.0005_lambda${i}_swpTrue_PE16_sawb_tanh_pactTrueA4_resumeTrue_fflf \
        --epochs ${epochs}  --learning_rate  0.01 \
        --optimizer ${optimizer} \
        --schedule 80 120 160    --gammas 0.1 0.1 0.5    \
        --batch_size ${batch_size} --workers 4 --ngpu 1 --gpu_id 2 \
        --print_freq 100 --decay 0.0005 \
        --lamda ${i}   --ratio 0.7 \
        --resume ${pretrained_model} \
        --clp \
        --a_lambda 0.01 \
        --fine_tune \
        --swp \
done
