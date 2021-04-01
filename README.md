# campustown - ramen 
python main.py campus campus_LeNet ../log/tofu_test ../data/두부\ 데이터셋 --data_load True --objective one-class --lr 0.0001 --n_epochs 150 --lr_milestone 50 --batch_size 64 --weight_decay 0.5e-6 --pretrain True --ae_lr 0.0001 --ae_n_epochs 150 --ae_lr_milestone 50 --ae_batch_size 64 --ae_weight_decay 0.5e-3 ;

# tofu 
python main.py campus campus_LeNet ../log/tofu_test ../data/두부\ 데이터셋 --data_load True --objective one-class --pretrain False --load_model /home/yeong95/svdd/deep-svdd-campus_town/log/tofu_test/model.tar