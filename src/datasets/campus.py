from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

# 경고 메시지 무시하기
import warnings
warnings.filterwarnings("ignore")

plt.ion()   # 반응형 모드

# load data
data_path = r'/home/yeong95/svdd/Deep-SVDD-PyTorch/data/라면 데이터/라면_이미지_640'
train_name = 'train_img.npy'
train_data_name = os.path.join(data_path,train_name)
train_data = np.load(train_data_name)
 

class Campustown_Dataset(Dataset):
    
    def __init__(self, npy_file, root_dir, transform=None):
        """
        Args:
            npy_file (string): npy 파일의 경로
            root_dir (string): 모든 이미지가 존재하는 디렉토리 경로
            transform (callable, optional): 샘플에 적용될 Optional transform
        """
        self.landmarks_frame = np.load(npy_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.landmarks_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir,
                                self.landmarks_frame.iloc[idx, 0])
        image = io.imread(img_name)
        landmarks = self.landmarks_frame.iloc[idx, 1:]
        landmarks = np.array([landmarks])
        landmarks = landmarks.astype('float').reshape(-1, 2)
        sample = {'image': image, 'landmarks': landmarks}

        if self.transform:
            sample = self.transform(sample)

        return sample

campus_dataset = Campustown_Dataset(npy_file=train_data_name,
                                    root_dir=data_path)


