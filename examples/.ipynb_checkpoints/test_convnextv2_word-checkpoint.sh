#!/bin/bash

# 移动到上一级目录
cd ..

if [ $epoch_time ]; then
    EPOCH_TIME=$epoch_time
else
    EPOCH_TIME=2000
fi

# 新增 max_itera 参数
if [ $max_itera ]; then
    MAX_ITERA=$max_itera
else
    MAX_ITERA=30000
fi

if [ $out_dir ]; then
    OUT_DIR=$out_dir
else
    OUT_DIR='./model_out_word_convnext'
fi

if [ $cfg ]; then
    CFG=$cfg
else
    CFG='configs/convnextv2_unet.yaml'
fi

if [ $data_dir ]; then
    DATA_DIR=$data_dir
else
    DATA_DIR='./data/data'
fi

if [ $learning_rate ]; then
    LEARNING_RATE=$learning_rate
else
    LEARNING_RATE=0.001
fi

if [ $img_size ]; then
    IMG_SIZE=$img_size
else
    IMG_SIZE=224
fi

if [ $batch_size ]; then
    BATCH_SIZE=$batch_size
else
    BATCH_SIZE=24
fi

# 新增设备选项
if [ $device ]; then
    DEVICE=$device
else
    DEVICE='cuda:1'
fi

echo "start test model"
# python test_mlla.py --dataset word --cfg $CFG --volume_path $DATA_DIR --max_epochs $EPOCH_TIME --output_dir $OUT_DIR --img_size $IMG_SIZE --base_lr $LEARNING_RATE --batch_size $BATCH_SIZE --device $DEVICE
python test_convnextv2.py --dataset word --cfg $CFG --is_saveni --volume_path $DATA_DIR --max_epochs $EPOCH_TIME --output_dir $OUT_DIR --img_size $IMG_SIZE --base_lr $LEARNING_RATE --batch_size $BATCH_SIZE --device $DEVICE --max_iterations $MAX_ITERA