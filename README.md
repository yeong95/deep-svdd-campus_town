## Instructions 
* Training Deep SVDD
    python main.py campus resnet ../log/tofu_test ./datasets --data_load True --objective one-class --lr 0.0005 --n_epochs 150 --lr_milestone 75 --batch_size 32 --weight_decay 0.5e-6 --pretrain True --ae_lr 0.001 --ae_n_epochs 500 --ae_lr_milestone 200 --ae_batch_size 8 --ae_weight_decay 0.5e-3 ;

using bash script 
    ./main.sh

* Load saved model and test
    python main.py campus resnet ../log/load_tofu_test ./datasets --data_load True --pretrain False --batch_size 32 --load_model '../log/tofu_test/model_tmp_saved/model_10.tar'

* Load several saved model and check which model has best accuracy 
comment out deep_SVDD.train/save_results/save_model in main.py 
    ./load_model_and_test.sh

* Plot t_sne
    python main.py campus resnet ../log/tofu_test ../data/"두부 데이터셋" --data_load True --objective one-class --pretrain False --load_model C:\Users\yeong95\svdd\log\tofu_test\model.tar




