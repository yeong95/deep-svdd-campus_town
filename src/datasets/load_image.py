import pandas as pd
import numpy as np
import os
from PIL import Image
import pickle


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
        os.chdir(r'/content/deep-svdd-campus_town/src') # reset to original path
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
            for i, file in enumerate(train_file_list):
                if i==500: break
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
    
    os.chdir(r'/home/yeong95/svdd/deep-svdd-campus_town/src') # reset to original path   
    
    return train_image, train_class, test_image, test_label, test_class


if __name__ == '__main__':
    data_path = r'/home/yeong95/svdd/deep-svdd-campus_town/data/두부 데이터셋'
    train_path = 'train'
    test_path = 'test'
    load = False
    train_image, train_class, test_image, test_label, test_class = train_test_numpy_load(data_path,train_path,test_path,load)