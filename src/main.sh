#!/bin/bash

# rm -r ../log/tofu_test/ ../log/load_tofu_test/
# mkdir ../log/tofu_test ../log/load_tofu_test
# python main_optuna.py campus resnet ../log/tofu_test ./datasets --data_load True --objective one-class \
# --lr 0.0005 --n_epochs 1 --lr_milestone 50 --batch_size 8 --weight_decay 0.5e-6 --pretrain False \
# --ae_lr 0.001 --ae_n_epochs 200 --ae_lr_milestone 200 --ae_batch_size 8 --ae_weight_decay 0.5e-3 \
# --device cuda;

# load model 
rm ../log/tofu_test/log.txt
python main_optuna.py campus resnet ../log/tofu_test ./datasets --data_load True --objective one-class \
--lr 0.0005 --n_epochs 150 --lr_milestone 50 --batch_size 16 --weight_decay 0.5e-6 --pretrain False \
--device cuda --load_model ../log/tofu_test/pretrained_model.tar;