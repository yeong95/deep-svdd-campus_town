from torch.utils.data import Subset
from PIL import Image
import sys
sys.path.append("/home/yeong95/svdd/deep-svdd-campus_town/src")
from base.torchvision_dataset import TorchvisionDataset
from preprocessing import get_target_label_idx, global_contrast_normalization

import torchvision.transforms as transforms

# load data
data_path = r'/home/yeong95/svdd/deep-svdd-campus_town/data/라면 데이터/라면_이미지_640'
data_name = 'train_img.npy'
train_data = np.load(os.path.join(data_path,data_name)).reshape(300,640,640)



class Campustown_Dataset(TorchvisionDataset):

    def __init__(self, root):
        super().__init__(root)

        self.n_classes = 2  # 0: normal, 1: outlier

        # Pre-computed min and max values (after applying GCN) from train data per class
        min_max = [(img_min, img_max)]

        # MNIST preprocessing: GCN (with L1 norm) and min-max feature scaling to [0,1]
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Lambda(lambda x: global_contrast_normalization(x, scale='l1')),
                                        transforms.Normalize([min_max[0],
                                                              min_max[1]-min_max[0]])
        
        self.train_set = MyCampus(root=self.root, train=True, transform=transform)
        self.test_set = MyCampus(root=self.root, train=False, transform=transform)
    
class MyCampus():
    def __init__(self, *args, **kwargs):
        super(MyCampus, self).__init__(*args, **kwargs)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            triple: (image, target, index) where target is index of the target class.
        """
        os.path.join(root, )
        self.train_data = 
        if self.train:
            img, target = self.train_data[index], 0
        else:
            img, target = self.test_data[index], self.test_labels[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img.numpy(), mode='L')

        if self.transform is not None:
            img = self.transform(img)


        return img, target, index  # only line changed


campus_dataset = Campustown_Dataset(files_in_train, data_path)




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
