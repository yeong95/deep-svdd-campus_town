# campustown - ramen 
# train from scratch
    python main.py campus resnet ../log/campus_test ../data/라면\ 데이터/라면_이미지_640 --lr 0.0005 --n_epochs 150 --lr_milestone 50 --batch_size 128 --weight_decay 0.5e-6 --pretrain True --ae_lr 0.001 --ae_n_epochs 150 --ae_lr_milestone 50 --ae_batch_size 200 --ae_weight_decay 0.5e-3;

# load model and train
    python main.py campus campus_LeNet ../log/campus_test ../data/라면\ 데이터/라면_이미지_640 --lr 0.0001 --n_epochs 150 --lr_milestone 50 --batch_size 128 --weight_decay 0.5e-6 --pretrain False --ae_lr 0.0001 --ae_n_epochs 150 --ae_lr_milestone 50 --ae_batch_size 200 --ae_weight_decay 0.5e-3 --load_model r"/home/yeong95/svdd/deep-svdd-campus_town/log/campus_test/model.tar";
    
