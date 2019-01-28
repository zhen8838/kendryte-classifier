#!/bin/bash

python3 train_softmax.py \
--model_def models.mobilenet_v1 \
--data_dir /media/zqh/Datas/DataSet/flower_photos \
--pretrained_model "../pretrained/mobilenetv1_1.0.pkl" \
--gpus 0 \
--image_size 224 \
--logs_base_dir backup_classifier \
--models_base_dir backup_classifier \
--batch_size 16 \
--epoch_size 200 \
--learning_rate 0.0004 \
--max_nrof_epochs 5 \
--class_num 5 \
--use_fixed_image_standardization \
--optimizer ADAM \
--keep_probability 1.0 \
--learning_rate_decay_epochs 10 \
--learning_rate_decay_factor 0.9