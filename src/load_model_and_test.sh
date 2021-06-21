#!/bin/bash
for var in {10..150..10}
do
    model_path=../log/tofu_test/model_tmp_saved/model_"$var".tar
    python main.py campus resnet ../log/load_tofu_test ./datasets --data_load True --pretrain False --batch_size 32 --load_model $model_path
done