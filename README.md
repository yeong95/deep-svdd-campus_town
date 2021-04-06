# campustown - ramen 
python main.py campus campus_LeNet ../log/tofu_test ../data/두부\ 데이터셋 --data_load True --objective one-class --lr 0.0001 --n_epochs 150 --lr_milestone 50 --batch_size 64 --weight_decay 0.5e-6 --pretrain True --ae_lr 0.0001 --ae_n_epochs 150 --ae_lr_milestone 50 --ae_batch_size 64 --ae_weight_decay 0.5e-3 ;

# tofu - training
python main.py campus resnet ../log/tofu_test ../data/"두부 데이터셋" --data_load True --objective one-class --lr 0.0001 --n_epochs 150 --lr_milestone 50 --batch_size 128 --weight_decay 0.5e-6 --pretrain True --ae_lr 0.0001 --ae_n_epochs 150 --ae_lr_milestone 50 --ae_batch_size 64 --ae_weight_decay 0.5e-3 ;

# tofu - plot t_sne
python main.py campus campus_LeNet ../log/tofu_test ../data/"두부 데이터셋" --data_load True --objective one-class --pretrain False --load_model C:\Users\yeong95\svdd\log\tofu_test\model.tar

