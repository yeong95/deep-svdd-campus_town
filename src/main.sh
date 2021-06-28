#!/bin/bash

#rm ../log/tofu_test/* ../log/load_tofu_test/*
python main.py campus resnet ../log/tofu_test ./datasets --data_load True --objective one-class --lr 0.0005 --n_epochs 150 --lr_milestone 75 --batch_size 16 --weight_decay 0.5e-6 --pretrain True --ae_lr 0.001 --ae_n_epochs 150 --ae_lr_milestone 200 --ae_batch_size 8 --ae_weight_decay 0.5e-3 --device cuda:0;