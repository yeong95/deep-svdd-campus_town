import pandas as pd
import numpy as np
import os
from PIL import Image
import pickle
# def train_test_numpy_load(data_path,train_path,test_path):
#     os.chdir(os.path.join(data_path,train_path))
#     folders = [name for name in os.listdir(".") if os.path.isdir(name)]
#     train_file_frame = pd.DataFrame(columns=['file_list', 'label', 'class'])
#     test_file_frame = pd.DataFrame(columns=['file_list', 'label', 'class'])
#     for fold in folders:
#         train_tmp = pd.DataFrame(columns=train_file_frame.columns)
#         test_tmp = pd.DataFrame(columns=test_file_frame.columns)
#         train_file_list = os.listdir(os.path.join(data_path,train_path,fold))
#         test_file_list = os.listdir(os.path.join(data_path,test_path,fold))
#         if fold in ['정상A', '정상B']:
#             train_label = [0]*len(train_file_list)
#             test_label = [0]*len(test_file_list)
#         else:
#             train_label = [1]*len(train_file_list)
#             test_label = [1]*len(test_file_list)
#         train_tmp['file_list'] = train_file_list
#         test_tmp['file_list'] = test_file_list
#         train_tmp['label'] = train_label
#         test_tmp['label'] = test_label
#         train_tmp['class'] = fold
#         test_tmp['class'] = fold
        
#         train_file_frame = train_file_frame.append(train_tmp, ignore_index=True)
#         test_file_frame = test_file_frame.append(test_tmp, ignore_index=True)
    
#     # train image -> numpy 
#     train_image = [] 
#     train_class = []
#     normal_index = np.logical_or((train_file_frame['class'] =='정상A'), (train_file_frame['class'] =='정상B'))
#     train = train_file_frame[normal_index]
#     for i in range(len(train)):
#         file = train.iloc[i][0]
#         img=os.path.join(data_path,train_path,train.iloc[i][2],file)
#         image = np.asarray(Image.open(img))
#         train_image.append(image)
#         train_class.append(train.iloc[i][2])
#     train_image = np.array(train_image)
    
#     # test image -> numpy
#     test_image = []
#     test_label = []
#     test_class = []
#     for i in range(len(test_file_frame)):
#         file = test_file_frame.iloc[i][0]
#         img = os.path.join(data_path,test_path,test_file_frame.iloc[i][2],file)
#         image = np.asarray(Image.open(img))
#         test_image.append(image)
#         test_label.append(test_file_frame.iloc[i][1])
#         test_class.append(test_file_frame.iloc[i][2])
#     test_image = np.array(test_image)
#     test_label = np.array(test_label)
        
#     return train_image, train_class, test_image, test_label, test_class

def train_test_numpy_load(data_path,train_path,test_path,load):
    
    if load:
        os.chdir(data_path)
        with open('train_image.pickle', 'rb') as f:
            train_image = pickle.load(f)
        with open('train_class.pickle', 'rb') as f:
            train_class = pickle.load(f)
        with open('test_image.pickle', 'rb') as f:
            test_image = pickle.load(f)
        with open('test_label.pickle', 'rb') as f:
            test_label = pickle.load(f)
        with open('test_class.pickle', 'rb') as f:
            test_class = pickle.load(f)
        return train_image, train_class, test_image, test_label, test_class 
    
    os.chdir(os.path.join(data_path,train_path))
    folders = [name for name in os.listdir(".") if os.path.isdir(name)]
    
    train_image = [] 
    train_class = []    
    test_image = []
    test_label = []
    test_class = []
    
    for fold in folders:
        train_file_list = os.listdir(os.path.join(data_path,train_path,fold))
        test_file_list = os.listdir(os.path.join(data_path,test_path,fold))
        
        
        if fold in ['정상A', '정상B']:
            for file in train_file_list:
                img = os.path.join(data_path,train_path,fold,file)
                image = np.asarray(Image.open(img).resize((640,640)))
                train_image.append(image)
            train_class += [fold]*len(train_file_list)               
            label = 0
        else:
            label = 1             

        for file in test_file_list:
            img = os.path.join(data_path,test_path,fold,file)
            image = np.asarray(Image.open(img).resize((640,640)))
            test_image.append(image)
        test_class += [fold]*len(test_file_list)
        test_label += [label]*len(test_file_list)   
        
    train_image = np.array(train_image)
    test_image = np.array(test_image)
    test_label = np.array(test_label)
    
    os.chdir(data_path)
    with open('train_image.pickle','wb') as f:
        pickle.dump(train_image,f,pickle.HIGHEST_PROTOCOL)
    with open('train_class.pickle','wb') as f:
        pickle.dump(train_class,f,pickle.HIGHEST_PROTOCOL)
    with open('test_image.pickle','wb') as f:
        pickle.dump(test_image,f,pickle.HIGHEST_PROTOCOL)
    with open('test_label.pickle','wb') as f:
        pickle.dump(test_label,f,pickle.HIGHEST_PROTOCOL)
    with open('test_class.pickle','wb') as f:
        pickle.dump(test_class,f,pickle.HIGHEST_PROTOCOL)   
    
    return train_image, train_class, test_image, test_label, test_class


if __name__ == '__main__':
    data_path = r'/home/yeong95/svdd/deep-svdd-campus_town/data/두부 데이터셋'
    train_path = 'train'
    test_path = 'test'
    load = False
    train_image, train_class, test_image, test_label, test_class = train_test_numpy_load(data_path,train_path,test_path,load)
    

            
    
        
        
        
    
    




