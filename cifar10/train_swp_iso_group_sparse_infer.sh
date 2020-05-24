#!/usr/bin/ sh

PYTHON="/home/mengjian/anaconda3/envs/neurosim_test/bin/python3"

############ directory to save result #############
DATE=`date +%Y-%m-%d`

if [ ! -d "$DIRECTORY" ]; then
    mkdir ./save
    mkdir ./dataset
    mkdir ./save/${DATE}/
fi

############ Configurations ###############
model=adc_resnet20
dataset=cifar10
batch_size=128
col_size=16
group_size=16
adc_precision=5

lambda_=0.001

pretrained_model="./save/resnet20/full_precsion/decay0.0002_w32_a32_fullprecision/model_best.pth.tar"
eval_model="./save/resnet20/iso_group_sparsity/ch16/skp_group16_4bit/decay0.0005_lambda0.001_alambda0.01_w4_a4_qsc_grp4_fl4bit_ll4bit/model_best.pth.tar"

$PYTHON -W ignore main_iso_group_sparse_infer.py --dataset ${dataset} \
    --data_path ./dataset/   \
    --arch ${model} --save_path ./save/resnet20/iso_group_sparsity/ch${group_ch}/skp_group16_4bit/decay${wd}_lambda${lambda_}_alambda${a_lambda}_w4_a4_qsc_grp4_fl4bit_ll4bit_eval \
    --batch_size ${batch_size} --workers 4 --ngpu 1 --gpu_id 2 \
    --resume ${eval_model} \
    --evaluate \
    --adc_infer \
    --col_size ${col_size} \
    --group_size ${group_size} \
    --ADCprecision ${adc_precision}

