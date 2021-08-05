#!/bin/bash

#############################
# normal  train
#############################
rm ../log/tofu_test/log*.txt  ../log/tofu_test/metric*.pickle ../log/tofu_test/confusion_matrix*.png
for var in {1..10}
do
    python main.py campus resnet ../log/tofu_test ./datasets --data_load True --objective one-class \
    --lr 0.00001 --n_epochs 50 --lr_milestone 50 --batch_size 16 --weight_decay 0.5e-6 --pretrain False \
    --device cuda:1 --load_model ../log/tofu_test/model.tar;
   
    python utils/confusion_matrix.py

    rm -r ../log/tofu_test/model_tmp_saved
    mv ../log/tofu_test/log.txt ../log/tofu_test/log$var.txt
    mv ../log/tofu_test/confusion_matrix.png ../log/tofu_test/confusion_matrix$var.png
    mv ../log/tofu_test/metric.pickle ../log/tofu_test/metric$var.pickle
    mv ../log/tofu_test/test_score.pickle ../log/tofu_test/test_score$var.pickle
done


#rm ../log/tofu_test/*.png ../log/tofu_test/*.txt ../log/tofu_test/*.json

