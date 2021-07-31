import pandas as pd
import numpy as np 
import os
from glob import glob

from sklearn.metrics import roc_curve
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score, recall_score, precision_score

import pickle

from PIL import Image



def test_load(data_path, test_path):
    normal_folders = ['정상A', '정상B']
    abnormal_folders = ['금속', '머리카락', '벌레', '상단불량D', '상단불량E', '유바', '탄화물', '플라스틱']

    tmp_list = []
    
    for fold in (normal_folders+abnormal_folders):

        if fold in normal_folders:
            fpattern = os.path.join(data_path,  f'{test_path}/OK/{fold}/*.jpg')
        else:
            fpattern = os.path.join(data_path,  f'{test_path}/NG/{fold}/*.jpg')
            
        fpath = sorted(glob(fpattern))
        tmp_list += [*map(lambda x:x[45:], fpath)]   # '정상A/2020-11-29-113900570845.jpg'
    
    return tmp_list




def cherry_pick_test_image(test_image, test_class, test_label, pred_label, test_name_list):


    img_array = np.empty((0,640,640,3), dtype=np.int8) # np.int8로 꼭 설정 (이미지)
    label_list = []
    name_list = []
    class_list = []


    for class_name in set(test_class):
        
        idx_ = (np.array(test_class)==class_name)
        y_pred = pred_label[idx_]
        y_true = test_label[idx_]        

        if class_name in ['정상A', '정상B']:
            img = test_image[idx_][y_pred==y_true][:50]
            label = [0]*50
            name_list += np.array(test_name_list)[idx_][y_pred==y_true][:50].tolist()
            print("class {} : {}" .format(class_name, img.shape[0]))
            class_list += [class_name] * 50 
        else:
            img = test_image[idx_][y_pred==y_true]
            if class_name in ['금속', '탄화물', '플라스틱', '상단불량E']:
                img = img[:12]
                label = [1]*12
                name_list += np.array(test_name_list)[idx_][y_pred==y_true][:12].tolist()
                class_list += [class_name] * 12
            else:
                img = img[:13]
                label = [1]*13
                name_list += np.array(test_name_list)[idx_][y_pred==y_true][:13].tolist()
                class_list += [class_name] * 13 
            
            print("class {} : {}" .format(class_name, img.shape[0]))
        
        img = img.astype(np.int8)
        img_array = np.append(img_array, img, axis=0)
        label_list += label    
    
        
    return img_array, np.array(label_list), class_list, name_list  



if __name__ == '__main__':

    test_score_path = r'/workspace/CAMPUS/CYK/campus/deep-svdd-campus_town/log/tofu_test'
    data_path = r'/workspace/CAMPUS/CYK/campus/deep-svdd-campus_town/src/datasets'
    with open(os.path.join(test_score_path,'test_score.pickle'), 'rb') as f:
        test_score = pickle.load(f)
    with open(os.path.join(data_path,'tripped_20_test_image.pickle'), 'rb') as f:
        test_image = pickle.load(f)
    with open(os.path.join(data_path,'tripped_20_test_label.pickle'), 'rb') as f:
        test_label = pickle.load(f)
    with open(os.path.join(data_path,'tripped_20_test_class.pickle'), 'rb') as f:
        test_class = pickle.load(f)

    indices, labels, scores = zip(*test_score)
    indices, labels, scores = np.array(indices), np.array(labels), np.array(scores)
    fpr, tpr, threshold = roc_curve(labels, scores)
    print("auc score: {}" .format(roc_auc_score(labels, scores)))

    # find optimal threshold (Youden’s J statistic.) -> TPR - FPR 
    best_index = np.argmax(tpr-fpr)
    best_threshold = threshold[best_index]
    # new_threshold = threshold[fpr==0.2][-1] # fpr는 0.2이고 그 중 tpr이 가장 높은 threshold 선정 
    pred_label = np.zeros(len(scores))
    pred_label[scores>best_threshold] = 1 # threshold보다 큰 것은 이상(label:1)
    print("fpr is {}" .format(fpr[best_index]))


    data_path = '/workspace/CAMPUS/TOFU_data_20p_trim'
    test_path = 'test'

    test_name_list = test_load(data_path, test_path)
    
    img_array, label_list, class_list, name_list = cherry_pick_test_image(test_image, test_class, test_label, pred_label, test_name_list)

    print(img_array.shape)
    print(img_array.dtype)
    print(len(label_list))
    print(len(class_list))
    print(len(name_list))


    with open('datasets/200_test_image.pickle','wb') as f:
        pickle.dump(img_array,f, protocol=4)
    with open('datasets/200_test_label.pickle','wb') as f:
        pickle.dump(label_list,f, protocol=4)
    with open('datasets/200_test_class.pickle','wb') as f:
        pickle.dump(class_list,f, protocol=4)
    with open('datasets/200_test_file_name.pickle','wb') as f:
        pickle.dump(name_list,f, protocol=4)

    # with open('datasets/200_test_image.pickle' , 'rb') as f:
    #     test_image = pickle.load(f)
    # with open('datasets/tripped_20_test_image.pickle', 'rb') as f:
    #     test_image2 = pickle.load(f)
    # import pdb;pdb.set_trace()