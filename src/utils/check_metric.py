import pandas as pd
import numpy as np
import os
import pickle

data_path = '/workspace/CAMPUS/CYK/campus/deep-svdd-campus_town/log/tofu_test'

df_ = pd.DataFrame()

acc_list = []
prec_list = []
recall_list = []

for i in range(1, 11):
    with open(os.path.join(data_path,'metric'+str(i)+'.pickle'), 'rb') as f:
        metric_dict = pickle.load(f)
    acc, precision, recall = metric_dict['accuracy'], metric_dict['precision'], metric_dict['recall']
    acc_list.append(acc)
    prec_list.append(precision)
    recall_list.append(recall)

df_['acc'] = acc_list
df_['precision'] = prec_list
df_['recall'] = recall_list

df_.describe()

import pdb;pdb.set_trace()