from __future__ import print_function
import time
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
%matplotlib inline
import matplotlib.pyplot as plt
import seaborn as sns
from deepSVDD import DeepSVDD

# path 
load_model = r'/home/yeong95/svdd/deep-svdd-campus_town/log/tofu_test/model.tar'

# load model 
deep_SVDD = DeepSVDD(objective='one-class', nu=0.1)
deep_SVDD.set_network('campus_LeNet')
deep_SVDD.load_model(model_path=load_model, load_ae=True)



tsne = TSNE(n_components=2)
TSNE.fit_transform()