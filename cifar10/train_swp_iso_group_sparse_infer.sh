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
model=vgg7_adc
dataset=cifar10
batch_size=1
col_size=16
group_size=16
adc_precision=5
cell_bit=2

lambda_=0.001

pretrained_model="./save/resnet20/full_precsion/decay0.0002_w32_a32_fullprecision/model_best.pth.tar"
# eval_model="./save/resnet20/w4_a4_quant_baseline_full_quant/decay0.0005_w4_a4_fullprecision_eval/model_best.pth.tar"
# eval_model="./save/resnet20/iso_group_sparsity/ch16/skp_group8_4bit/decay0.0005_lambda0.001_alambda0.01_w4_a4_qsc_grp8_fl4bit_ll4bit/model_best.pth.tar"
# eval_model="./save/resnet20/iso_group_sparsity/ch16/skp_group4_4bit/decay0.0005_lambda0.001_alambda0.01_w4_a4_qsc_grp4_fl4bit_ll4bit/model_best.pth.tar"

eval_model="./save/vgg7_bn_fuse/full_precision/decay0.0001_w4_a4/model_best.pth.tar"

$PYTHON -W ignore main_iso_group_sparse_infer.py --dataset ${dataset} \
    --data_path ./dataset/   \
    --arch ${model} --save_path ./save/vgg7_bn_fuse/full_precision/decay0.0001_w4_a4_eval \
    --batch_size ${batch_size} --workers 4 --ngpu 1 --gpu_id 2 \
    --resume ${eval_model} \
    --evaluate \
    --adc_infer \
    --col_size ${col_size} \
    --group_size ${group_size} \
    --ADCprecision ${adc_precision} \
    --cell_bit ${cell_bit} \
    --bn_fuse

