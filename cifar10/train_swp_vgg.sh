#!/usr/bin/ sh

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
iso_group=16

ub=0.002
lb=0.002

diff=0.001

# add more labels as additional info into the saving path
label_info=

pretrained_model="cifar10/save/vgg7_bnfalse/full_precision/decay0.0001_w32_a32/model_best.pth.tar"
# eval_model="./save/vgg7/ch16/group16/decay0.0005_lambda0.003_w4_a4_swpTrue_resumeTrue_symm/model_best.pth.tar"
# eval_model="./save/vgg7/ch16/group16/decay0.0005_lambda0.003_w4_a4_swpTrue_resumeTrue_symm/model_best.pth.tar"

for i in $(seq ${lb} ${diff} ${ub})
do
    $PYTHON -W ignore main_iso_group_sparse.py --dataset ${dataset} \
        --data_path ./dataset/   \
        --arch ${model} --save_path ./save/vgg7_bnfalse/ch${group_ch}/group${iso_group}/decay0.0005_lambda${i}_w4_a4_swpTrue_resumeTrue_symm \
        --epochs ${epochs}  --learning_rate  0.01 \
        --optimizer ${optimizer} \
        --schedule 80 120   --gammas 0.1 0.1 \
        --batch_size ${batch_size} --workers 4 --ngpu 1 --gpu_id 2 \
        --print_freq 100 --decay 0.0005 \
        --lamda ${i}   --ratio 0.7 \
        --resume ${eval_model} \
        --fine_tune \
        --clp \
        --a_lambda 0.001 \
        --swp \
	    --group_ch ${group_ch}
done
