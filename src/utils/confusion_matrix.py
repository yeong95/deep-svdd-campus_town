import pickle
import pandas as pd 
import numpy as np 
import os

from sklearn.metrics import roc_curve
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
# %matplotlib inline 
import scikitplot as skplt 
from PIL import Image
from sklearn.metrics import roc_auc_score, recall_score, precision_score
import sys




# plot confusion matrix
def plot_confusion_matrix(test_label,pred_label,test_score_path):
    skplt.metrics.plot_confusion_matrix(y_true=test_label,y_pred=pred_label)
    plt.savefig(os.path.join(test_score_path, 'confusion_matrix.png'))
    accuracy = accuracy_score(test_label, pred_label)
    recall = recall_score (test_label, pred_label)
    precision = precision_score(test_label, pred_label)
    print("accuracy score: {}" .format(accuracy))
    print("recall: {}" .format(recall))
    print("preicision: {}" .format(precision))

    return accuracy, recall, precision




# 클래스별 hit ratio(recall)
def class_recall_and_save_wrong_images(test_image, test_class,test_label,pred_label,wrong_save_path, true_save_path):
    recall_dict = dict()
    for class_name in set(test_class):
        print("class: {}" .format(class_name))
        idx_ = (np.array(test_class)==class_name)
        y_pred = pred_label[idx_]
        y_true = test_label[idx_]        
        # print("{} total count : {}" .format(class_name, len(y_true)))
        # if class_name in ['정상B','머리카락','유바']:
        #     # save wrong prediction image
        #     class_image = test_image[idx_][y_pred!=y_true]
        #     for i, image in enumerate(class_image):
        #         img = Image.fromarray(image)
        #         img.save(os.path.join(wrong_save_path,(class_name+"_"+str(i)+".png")))

        #     # save true prediction image
        #     class_image = test_image[idx_][y_pred==y_true]
        #     for i, image in enumerate(class_image):
        #         img = Image.fromarray(image)
        #         img.save(os.path.join(true_save_path,(class_name+"_"+str(i)+".png")))


        accuracy = accuracy_score(y_true, y_pred)
        recall_dict[class_name] = accuracy
    recall_df = pd.DataFrame()
    recall_df['class'] = recall_dict.keys()
    recall_df['recall'] = recall_dict.values()
    print(recall_df.sort_values(by=['recall']))
    


if __name__ == '__main__':
    
    test_score_path = r'/workspace/CAMPUS/CYK/campus/deep-svdd-campus_town/log/tofu_test'
    data_path = r'/workspace/CAMPUS/CYK/campus/deep-svdd-campus_town/src/datasets'
    with open(os.path.join(test_score_path,'test_score.pickle'), 'rb') as f:
        test_score = pickle.load(f)
    with open(os.path.join(data_path,'200_test_image.pickle'), 'rb') as f:
        test_image = pickle.load(f)
    with open(os.path.join(data_path,'200_test_label.pickle'), 'rb') as f:
        test_label = pickle.load(f)
    with open(os.path.join(data_path,'200_test_class.pickle'), 'rb') as f:
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
    
    accuracy_score, recall_score, precision_score = plot_confusion_matrix(test_label,pred_label,test_score_path)
    metric_dict = {}
    metric_dict['accuracy'] = accuracy_score
    metric_dict['recall'] = recall_score
    metric_dict['precision'] = precision_score
    with open(os.path.join(test_score_path, 'metric.pickle'), 'wb') as f:
        pickle.dump(metric_dict, f, protocol=4)

    
    sys.exit(0)    
    wrong_save_path = '/workspace/CAMPUS/CYK/campus/deep-svdd-campus_town/log/wrong_image_sample'
    true_save_path = '/workspace/CAMPUS/CYK/campus/deep-svdd-campus_town/log/true_image_sample'

    class_recall_and_save_wrong_images(test_image,test_class,test_label,pred_label,wrong_save_path,true_save_path)
