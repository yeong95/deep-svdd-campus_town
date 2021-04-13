import pandas as pd
import numpy as np
import os
from PIL import Image
import pickle5 as pickle

def train_test_numpy_load(data_path,train_path,test_path,load):
    if load:
        with open(os.path.join(data_path,'train_image.pickle'), 'rb') as f:
            train_image = pickle.load(f)
        with open(os.path.join(data_path,'test_image.pickle'), 'rb') as f:
            test_image = pickle.load(f)
        with open(os.path.join(data_path,'test_label.pickle'), 'rb') as f:
            test_label = pickle.load(f)
        return train_image, test_image, test_label
    train_file_list = os.listdir(os.path.join(data_path,train_path))
    test_file_list = os.listdir(os.path.join(data_path,test_path))
    tmp_list=[]
    for x in train_file_list:
        tmp=0
        for y in x[4:7]:
            try:
                tmp = tmp*10+int(y)
            except:
                pass
        tmp_list.append(tmp)
        
    tmp_list_test=[]
    for x in test_file_list:
        tmp=0
        for y in x[:3]:
            try:
                tmp = tmp*10+int(y)
            except:
                pass
        tmp_list_test.append(tmp)
    train_file_list = np.array(train_file_list)[np.argsort(tmp_list)]
    test_file_list = np.array(test_file_list)[np.argsort(tmp_list_test)]

    # train image -> numpy 
    train_image = []
    for file in train_file_list:
        img=os.path.join(data_path,train_path,file)
        image = np.asarray(Image.open(img))
        train_image.append(image)
    train_image = np.array(train_image)

    # test image -> numpy
    test_image = []
    for file in test_file_list:
        img = os.path.join(data_path,test_path,file)
        image = np.asarray(Image.open(img))
        test_image.append(image)
    test_image = np.array(test_image)

    # test label -> numpy
    test_label = np.zeros((140,1))
    test_label[:10] = 1
    test_label[20:30] = 1
    test_label[40:60] = 1

    with open(os.path.join(data_path,'train_image.pickle'),'wb') as f:
        pickle.dump(train_image,f,pickle.HIGHEST_PROTOCOL)
    with open(os.path.join(data_path,'test_image.pickle'),'wb') as f:
        pickle.dump(test_image,f,pickle.HIGHEST_PROTOCOL)
    with open(os.path.join(data_path,'test_label.pickle'),'wb') as f:
        pickle.dump(test_label,f,pickle.HIGHEST_PROTOCOL)
    
    return train_image, test_image, test_label



