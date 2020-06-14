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
model=resnet20_lq
dataset=cifar10
epochs=200
batch_size=128
optimizer=SGD
group_ch=16

ub=0.002
lb=0.002
diff=0.001

# add more labels as additional info into the saving path
label_info=

pretrained_model="./save/resnet20/w4_a4_quant_baseline/decay0.0005_w4_a4_fullprecision/model_best.pth.tar"

for i in $(seq ${lb} ${diff} ${ub})
do
    $PYTHON -W ignore main_LQ.py --dataset ${dataset} \
        --data_path ./dataset/   \
        --arch ${model} --save_path ./save/resnet20/learnable_quant/ch${group_ch}/decay0.0005_lambda${i}_w4_a4_swpTrue_resumeTrue_qsc_symm_from_quant_pact_w \
        --epochs ${epochs}  --learning_rate  0.01 \
        --optimizer ${optimizer} \
        --schedule 80 120 160   --gammas 0.1 0.1 0.5 \
        --batch_size ${batch_size} --workers 4 --ngpu 1 --gpu_id 2 \
        --print_freq 100 --decay 0.0005 \
        --lamda ${i}   --ratio 0.7 \
        --resume ${pretrained_model} \
        --clp \
        --a_lambda 0.001 \
        --w_clp \
        --b_lambda 0.0005 \
        --fine_tune \
        --swp \
	    --group_ch ${group_ch}
done
