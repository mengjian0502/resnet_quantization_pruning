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
epochs=60
batch_size=128
optimizer=SGD
group_ch=16

ub=0.0005
lb=0.0005
diff=0.0002

# add more labels as additional info into the saving path
label_info=

# pretrained_model="./save/resnet20/decay0.0002_fullprecision_multiplecheckpoints_fflf/model_best.pth.tar"
pretrained_model="./save/resnet20/grp_sweep/ch16/decay0.0005_lambda0.0005_w4_a4_swpTrue_resumeTrue/model_best.pth.tar"

for i in $(seq ${lb} ${diff} ${ub})
do
    $PYTHON -W ignore main.py --dataset ${dataset} \
        --data_path ./dataset/   \
        --arch ${model} --save_path ./save/resnet20/grp_sweep/ch${group_ch}/decay0.0005_lambda${i}_w4_a4_swpTrue_resumeTrue_postquant \
        --epochs ${epochs}  --learning_rate  0.01 \
        --optimizer ${optimizer} \
        --schedule 30 40 45  --gammas 0.1 0.1 0.1 \
        --batch_size ${batch_size} --workers 4 --ngpu 1 --gpu_id 2 \
        --print_freq 100 --decay 0.0005 \
        --lamda ${i}   --ratio 0.7 \
        --resume ${pretrained_model} \
        --clp \
        --a_lambda 0.01 \
        --fine_tune
        # --swp \
	    # --group_ch ${group_ch}
done
