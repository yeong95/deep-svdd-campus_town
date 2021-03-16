from torch.utils.data import Subset
from PIL import Image
import sys
sys.path.append("/home/yeong95/svdd/deep-svdd-campus_town/src")
from base.torchvision_dataset import TorchvisionDataset
from .preprocessing import get_target_label_idx, global_contrast_normalization
import numpy as np
import torchvision.transforms as transforms
import torch
from torch.utils.data import Dataset
import os 


data_path = r'/home/yeong95/svdd/deep-svdd-campus_town/data/두부 데이터셋'
with open(os.path.join(data_path,'train_image.pickle'), 'rb') as f:
    train_data = pickle.load(f)

compute min-max
min_ = 10000
max_ = -10000
for data in train_data:
    torch_data = global_contrast_normalization(torch.from_numpy(data), 'l1')
    min_tmp = torch.min(torch_data)
    max_tmp = torch.max(torch_data)
    if min_ >= min_tmp:
        min_= min_tmp
    if max_ <= max_tmp:
        max_ = max_tmp
   
class Campustown_Dataset(TorchvisionDataset):

    def __init__(self, root, traindata=None, testdata=None, testlabel=None):
        super().__init__(root)

        self.n_classes = 2  # 0: normal, 1: outlier

        # Pre-computed min and max values (after applying GCN) from train data per class
        min_max = [(-3.9348, 0.7229)]

        # preprocessing GCN (with L1 norm) and min-max feature scaling to [0,1]
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Lambda(lambda x: global_contrast_normalization(x, scale='l1')),
                                        transforms.Normalize([min_max[0][0]],
                                                              [min_max[0][1]-min_max[0][0]])])
        
        self.train_set = MyCampus(train=True, transform=transform, train_data=traindata, test_data=testdata, test_label=testlabel)
        self.test_set = MyCampus(train=False, transform=transform, train_data=traindata, test_data=testdata, test_label=testlabel)
    
class MyCampus(Dataset):
    def __init__(self, train, transform=None, train_data=None, test_data=None, test_label=None):
        self.train = train
        self.train_data = torch.from_numpy(train_data)
        self.test_data = torch.from_numpy(test_data)
        self.test_label = torch.from_numpy(test_label)
        self.transform = transform
    def __len__(self):
        if self.train:
            return len(self.train_data)
        else:
            return len(self.test_data)
    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            triple: (image, target, index) where target is index of the target class.
        """
        
        if self.train:
            img, target = self.train_data[index], 0
        else:
            img, target = self.test_data[index], self.test_label[index][0]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img.numpy(), mode='L')

        if self.transform is not None:
            img = self.transform(img)
            
        return img, target, index  # only line changed




# campus_dataset = Campustown_Dataset(data_path, train_data, test_data, test_label)

# show sample image
# for i in range(len(campus_dataset)):
#     sample = campus_dataset[i]

# ax = plt.subplot(1, 4, i + 1)
# plt.tight_layout()
# ax.set_title('Sample #{}'.format(i))
# ax.axis('off')
# io.imshow(sample)
# plt.show()

# if i == 3:
#     plt.show()
#     break
