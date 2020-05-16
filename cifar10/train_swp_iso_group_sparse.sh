#!/usr/bin/ sh

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
epochs=2
batch_size=128
optimizer=SGD
group_ch=16

# coef=0.05
# TD_alpha_final=1.0
# TD_alpha=1.0

wd=0.0005
a_lambda=0.01

ub=0.001
lb=0.001
diff=0.001

pretrained_model="./save/resnet20/decay0.0002_fullprecision_multiplecheckpoints_fflf/model_best.pth.tar"
eval_model="./save/resnet20/iso_group_sparsity/ch16/skp_group8_4bit/decay0.0005_lambda0.001_alambda0.01_w4_a4_swpTrue_resumeTrue_qsc4bit_grp8_tmp/checkpoint.pth.tar"

for i in $(seq ${lb} ${diff} ${ub})
do
    $PYTHON -W ignore main_iso_group_sparse.py --dataset ${dataset} \
        --data_path ./dataset/   \
        --arch ${model} --save_path ./save/resnet20/iso_group_sparsity/ch${group_ch}/skp_group8_4bit/decay${wd}_lambda${i}_alambda${a_lambda}_w4_a4_swpTrue_resumeTrue_qsc4bit_grp8_eval \
        --epochs ${epochs}  --learning_rate  0.01 \
        --optimizer ${optimizer} \
        --schedule 80 120 160   --gammas 0.1 0.1 0.5 \
        --batch_size ${batch_size} --workers 4 --ngpu 1 --gpu_id 2 \
        --print_freq 100 --decay ${wd} \
        --lamda ${i}   --ratio 0.7 \
        --resume ${eval_model} \
        --clp \
        --a_lambda ${a_lambda} \
        --fine_tune \
        --group_ch ${group_ch} \
        --swp \
        --evaluate 
done
