#!/bin/bash

cd ..

if [ $epoch_time ]; then
    EPOCH_TIME=$epoch_time
else
    EPOCH_TIME=500
fi

if [ $out_dir ]; then
    OUT_DIR=$out_dir
else
    OUT_DIR='./model_out_Symmetry_ACDC'
fi

if [ $cfg ]; then
    CFG=$cfg
else
    CFG='configs/mlla_t.yaml'
fi

if [ $data_dir ]; then
    DATA_DIR=$data_dir
else
    DATA_DIR='data/ACDC'
fi

if [ $learning_rate ]; then
    LEARNING_RATE=$learning_rate
else
    LEARNING_RATE=0.0001
fi

if [ $img_size ]; then
    IMG_SIZE=$img_size
else
    IMG_SIZE=224
fi

if [ $batch_size ]; then
    BATCH_SIZE=$batch_size
else
    BATCH_SIZE=48
fi

# 新增设备选项
if [ $device ]; then
    DEVICE=$device
else
    DEVICE='cuda:1'
fi

echo "start train model"
python train_mlla_acdc.py --dataset ACDC --cfg $CFG --root_path $DATA_DIR --max_epochs $EPOCH_TIME --output_dir $OUT_DIR --img_size $IMG_SIZE --base_lr $LEARNING_RATE --batch_size $BATCH_SIZE --device $DEVICE
