import numpy as np
import os
from PIL import Image
import pickle
from glob import glob
from tqdm import tqdm



class tripped_train_test_numpy_load:
    def __init__(self, data_path, train_path, test_path, saved_path, name):
        self.data_path = data_path
        self.train_path = train_path
        self.test_path = test_path
        self.saved_path = saved_path
        self.data_name = name
        
        
    def train_load(self):
        folders = ['정상A', '정상B']
        train_image = []
        train_class = []
        
        for fold in folders:
            fpattern = os.path.join(self.data_path,  f'{self.train_path}/{fold}/*.jpg')
            fpath = sorted(glob(fpattern))
            for file in tqdm(fpath):
                image = np.asarray(Image.open(file))
                train_image.append(image)
            train_class += [fold]*len(fpath) 
            
        train_image = np.array(train_image)
        
        with open(self.data_name+'_train_image.pickle','wb') as f:
            pickle.dump(train_image,f, protocol=4)
        with open(self.data_name+'_train_class.pickle','wb') as f:
            pickle.dump(train_class,f, protocol=4)
    
    def test_load(self):
        normal_folders = ['정상A', '정상B']
        abnormal_folders = ['금속', '머리카락', '벌레', '상단불량D', '상단불량E', '유바', '탄화물', '플라스틱']
        test_image = []
        test_label = []
        test_class = []
        
        for fold in (normal_folders+abnormal_folders):
            fpattern = os.path.join(self.data_path,  f'{self.test_path}/{fold}/*.jpg')
            fpath = sorted(glob(fpattern))
            for file in tqdm(fpath):
                image = np.asarray(Image.open(file))
                test_image.append(image)
            if fold in ['정상A', '정상B']:
                label = 0
            else:
                label = 1
            test_class += [fold]*len(fpath)
            test_label += [label]*len(fpath) 
            # print("{} class : {}" .format(fold, len(test_file_list)))
            
        test_image = np.array(test_image)
        test_label = np.array(test_label)
        print(test_image.shape)
        
        with open(self.data_name+'_valid_image.pickle','wb') as f:
            pickle.dump(test_image,f, protocol=4)
        with open(self.data_name+'_valid_label.pickle','wb') as f:
            pickle.dump(test_label,f, protocol=4)
        with open(self.data_name+'_valid_class.pickle','wb') as f:
            pickle.dump(test_class,f, protocol=4) 
    
    def load(self):
        with open(os.path.join(self.saved_path,self.data_name+'_train_image.pickle'), 'rb') as f:
            train_image = pickle.load(f)
            index_ = np.random.choice(3666, 1000, replace=False) 
            train_image = train_image[index_]
        with open(os.path.join(self.saved_path,self.data_name+'_train_class.pickle'), 'rb') as f:
            train_class = pickle.load(f)
        with open(os.path.join(self.saved_path,self.data_name+'_valid_image.pickle'), 'rb') as f:
            valid_image = pickle.load(f)
        with open(os.path.join(self.saved_path,self.data_name+'_valid_label.pickle'), 'rb') as f:
            valid_label = pickle.load(f)
        with open(os.path.join(self.saved_path,self.data_name+'_valid_class.pickle'), 'rb') as f:
            valid_class = pickle.load(f)
        with open(os.path.join(self.saved_path,'200_test_image.pickle'), 'rb') as f:
            test_image = pickle.load(f)
        with open(os.path.join(self.saved_path,'200_test_label.pickle'), 'rb') as f:
            test_label = pickle.load(f)        
        with open(os.path.join(self.saved_path,'200_test_class.pickle'), 'rb') as f:
            test_class = pickle.load(f)
            
            
        return train_image, train_class, valid_image, valid_label, valid_class, test_image, test_label, test_class 



if __name__ == '__main__':
    data_path = '/workspace/CAMPUS/TOFU_BOX_margin_60/'
    train_path = 'train'
    test_path = 'valid'
    saved_path = '/workspace/CAMPUS/CYK/campus/deep-svdd-campus_town/src/datasets'
    name = 'Box_margin_60'
    data_load = tripped_train_test_numpy_load(data_path,train_path,test_path,saved_path,name)
    # data_load.train_load()
    # data_load.test_load()
    train_image, train_class, valid_image, valid_label, valid_class,\
     test_image, test_label, test_class = data_load.load()
    print("train image shape: {}" .format(train_image.shape))
    print("train class length: {}" .format(len(train_class)))
    print("valid image shape: {}" .format(valid_image.shape))
    print("valid class length: {}" .format(len(valid_class)))
    print("test image shape: {}" .format(test_image.shape))
    print("test class length: {}" .format(len(test_class)))
    import pdb;pdb.set_trace()