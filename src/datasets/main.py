from .mnist import MNIST_Dataset
from .cifar10 import CIFAR10_Dataset
from .campus import Campustown_Dataset


def load_dataset(dataset_name, data_path, normal_class):
    """Loads the dataset."""

    implemented_datasets = ('mnist', 'cifar10')
    assert dataset_name in implemented_datasets

    dataset = None

    if dataset_name == 'mnist':
        dataset = MNIST_Dataset(root=data_path, normal_class=normal_class)

    if dataset_name == 'cifar10':
        dataset = CIFAR10_Dataset(root=data_path, normal_class=normal_class)

    return dataset

def load_campus_dataset(datset_name, data_path, train_data, valid_data, valid_label, test_data, test_label):
    
    implemented_datasets = ('campus')
    assert datset_name in implemented_datasets
    
    dataset = None
    
    datset = Campustown_Dataset(data_path, train_data, valid_data, valid_label, test_data, test_label)

    return datset