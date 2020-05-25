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
model=vgg7
dataset=cifar10
epochs=200
batch_size=128
optimizer=SGD
group_ch=8

ub=0.001
lb=0.001
diff=0.001

# add more labels as additional info into the saving path
label_info=

pretrained_model="./save/vgg7/full_precision/decay0.0001_w32_a32/model_best.pth.tar"

for i in $(seq ${lb} ${diff} ${ub})
do
    $PYTHON -W ignore main.py --dataset ${dataset} \
        --data_path ./dataset/   \
        --arch ${model} --save_path ./save/resnet20/grp_sweep/ch${group_ch}/decay0.0005_lambda${i}_w4_a4_swpTrue_resumeTrue_qsc_symm_g03 \
        --epochs ${epochs}  --learning_rate  0.01 \
        --optimizer ${optimizer} \
        --schedule 80 120   --gammas 0.1 0.1 \
        --batch_size ${batch_size} --workers 4 --ngpu 1 --gpu_id 2 \
        --print_freq 100 --decay 0.0005 \
        --lamda ${i}   --ratio 0.7 \
        --resume ${pretrained_model} \
        --clp \
        --a_lambda 0.01 \
        --fine_tune \
        --swp \
	    --group_ch ${group_ch}
        --evalulate
done
