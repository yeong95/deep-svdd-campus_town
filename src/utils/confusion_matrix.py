import pickle
import pandas as pd 
import numpy as np 
import os
import pickle
from sklearn.metrics import roc_curve
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
# %matplotlib inline 
import scikitplot as skplt 
from PIL import Image

test_score_path = r'/workspace/CAMPUS/CYK/campus/deep-svdd-campus_town/log/tofu_test'
data_path = r'/workspace/CAMPUS/CYK/campus/deep-svdd-campus_town/src/datasets'
with open(os.path.join(test_score_path,'test_score.pickle'), 'rb') as f:
    test_score = pickle.load(f)
with open(os.path.join(data_path,'Box_test_image.pickle'), 'rb') as f:
    test_image = pickle.load(f)
with open(os.path.join(data_path,'Box_test_label.pickle'), 'rb') as f:
    test_label = pickle.load(f)
with open(os.path.join(data_path,'Box_test_class.pickle'), 'rb') as f:
    test_class = pickle.load(f)

indices, labels, scores = zip(*test_score)
indices, labels, scores = np.array(indices), np.array(labels), np.array(scores)
fpr, tpr, threshold = roc_curve(labels, scores)

# find optimal threshold (Youden’s J statistic.) -> TPR - FPR 
best_index = np.argmax(tpr-fpr)
import pdb;pdb.set_trace()
best_threshold = threshold[best_index]
# new_threshold = threshold[fpr==0.2][-1] # fpr는 0.2이고 그 중 tpr이 가장 높은 threshold 선정 
pred_label = np.zeros(len(scores))
pred_label[scores>best_threshold] = 1 # threshold보다 큰 것은 이상(label:1)

# plot confusion matrix
skplt.metrics.plot_confusion_matrix(y_true=test_label,y_pred=pred_label)
plt.savefig(os.path.join(test_score_path, 'confusion_matrix.png'))

# 클래스별 hit ratio(recall)
# import pdb;pdb.set_trace()
save_path = '/workspace/CAMPUS/CYK/campus/deep-svdd-campus_town/log/wrong_image_sample'
recall_dict = dict()
for class_name in set(test_class):
    idx_ = (np.array(test_class)==class_name)
    y_pred = pred_label[idx_]
    y_true = test_label[idx_]
    if class_name in ['머리카락', '정상A', '정상B', '벌레', '상단불량D']:
        class_image = test_image[idx_][y_pred!=y_true]
        for i, image in enumerate(class_image):
            img = Image.fromarray(image)
            img.save(os.path.join(save_path,(class_name+"_"+str(i)+".png")))
            if (i+1)==2: break
    accuracy = accuracy_score(y_true, y_pred)
    recall_dict[class_name] = accuracy

  
