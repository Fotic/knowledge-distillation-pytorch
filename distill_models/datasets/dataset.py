import os
import sys
import pickle
import random
import matplotlib.pyplot as plt
import numpy as np
import torch
import datasets.memd_dataset as memd
from torch.utils.data import Dataset, DataLoader
from datasets.base_dataset import BaseDataset
from datasets.dataset_config import dataset_config
import torchvision.transforms as transforms
import torchvision


def get_dataset(data_path, train_transforms, test_transforms, validation, shuffle_classes, class_order=None):
    all_data = {}
    path = data_path
    all_data['trn'] = {'x': [], 'y': []}
    all_data['val'] = {'x': [], 'y': []}
    all_data['tst'] = {'x': [], 'y': []}

    # read filenames and labels
    trn_lines = np.loadtxt(os.path.join(path, 'train.txt'), dtype=str)
    tst_lines = np.loadtxt(os.path.join(path, 'test.txt'), dtype=str)

    if class_order is None:
        num_classes = len(np.unique(trn_lines[:, 1]))
        class_order = list(range(num_classes))
    else:
        num_classes = len(class_order)
        class_order = class_order.copy()
    if shuffle_classes:
        np.random.shuffle(class_order)

    for this_image, this_label in trn_lines:
        if not os.path.isabs(this_image):
            this_image = os.path.join(path, this_image)
        this_label = int(this_label)
        if this_label not in class_order:
            continue
        # If shuffling is false, it won't change the class number
        this_label = class_order.index(this_label)

        # add it to the corresponding split
        all_data['trn']['x'].append(this_image)
        all_data['trn']['y'].append(this_label)

    for this_image, this_label in tst_lines:
        if not os.path.isabs(this_image):
            this_image = os.path.join(path, this_image)
        this_label = int(this_label)
        if this_label not in class_order:
            continue
        # If shuffling is false, it won't change the class number
        this_label = class_order.index(this_label)

        # add it to the corresponding split
        all_data['tst']['x'].append(this_image)
        all_data['tst']['y'].append(this_label)

    all_data['ncla'] = len(np.unique(all_data['trn']['y']))

    # validation
    if validation > 0.0:
        for cc in range(all_data['ncla']):
            cls_idx = list(np.where(np.asarray(all_data['trn']['y']) == cc)[0])
            rnd_img = random.sample(cls_idx, int(np.round(len(cls_idx) * validation)))
            rnd_img.sort(reverse=True)
            for ii in range(len(rnd_img)):
                all_data['val']['x'].append(all_data['trn']['x'][rnd_img[ii]])
                all_data['val']['y'].append(all_data['trn']['y'][rnd_img[ii]])
                all_data['trn']['x'].pop(rnd_img[ii])
                all_data['trn']['y'].pop(rnd_img[ii])

    train_dataset = BaseDataset(all_data['trn'], train_transforms)
    test_dataset = BaseDataset(all_data['tst'], test_transforms)
    valid_dataset = BaseDataset(all_data['val'], test_transforms)

    return train_dataset, test_dataset, valid_dataset, num_classes


def get_transforms(resize, pad, crop, flip, normalize, extend_channel):
    """Unpack transformations and apply to train or test splits"""

    trn_transform_list = []
    tst_transform_list = []

    # resize
    if resize is not None:
        # trn_transform_list.append(transforms.Resize(resize))
        # tst_transform_list.append(transforms.Resize(resize))
        # for CUB_200_2011
        trn_transform_list.append(transforms.Resize((resize, resize)))
        tst_transform_list.append(transforms.Resize((resize, resize)))

    # padding
    if pad is not None:
        trn_transform_list.append(transforms.Pad(pad))
        tst_transform_list.append(transforms.Pad(pad))

    # crop
    if crop is not None:
        trn_transform_list.append(transforms.RandomResizedCrop(crop))
        tst_transform_list.append(transforms.CenterCrop(crop))
        # for CUB_200_2011
        # trn_transform_list.append(transforms.RandomCrop(crop))
        # tst_transform_list.append(transforms.RandomCrop(crop))

    # flips
    if flip:
        trn_transform_list.append(transforms.RandomHorizontalFlip())

    # to tensor
    trn_transform_list.append(transforms.ToTensor())
    tst_transform_list.append(transforms.ToTensor())

    # normalization
    if normalize is not None:
        trn_transform_list.append(transforms.Normalize(mean=normalize[0], std=normalize[1]))
        tst_transform_list.append(transforms.Normalize(mean=normalize[0], std=normalize[1]))

    # gray to rgb
    if extend_channel is not None:
        trn_transform_list.append(transforms.Lambda(lambda x: x.repeat(extend_channel, 1, 1)))
        tst_transform_list.append(transforms.Lambda(lambda x: x.repeat(extend_channel, 1, 1)))

    return transforms.Compose(trn_transform_list), \
           transforms.Compose(tst_transform_list)


def get_loaders(dataset, num_workers, batch_size, validation):
    # get configuration for current dataset
    dc = dataset_config[dataset]
    data_path = dc['path']
    class_order = dc['class_order']
    # transformations
    trn_transform, tst_transform = get_transforms(resize=dc['resize'],
                                                  pad=dc['pad'],
                                                  crop=dc['crop'],
                                                  flip=dc['flip'],
                                                  normalize=dc['normalize'],
                                                  extend_channel=dc['extend_channel'])

    if dataset == "mnist":
        trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True,
                                              transform=trn_transform)
        num_classes = len(np.unique(trainset.targets))
        testset = torchvision.datasets.MNIST(root='./data', train=False, download=True,
                                             transform=tst_transform)
        trn_data = {'x': trainset.data.numpy(), 'y': trainset.targets.tolist()}
        tst_data = {'x': testset.data.numpy(), 'y': testset.targets.tolist()}
        all_data = memd.get_data(trn_data, tst_data, validation=validation, shuffle_classes=class_order is None,
                                 class_order=class_order)
        Dataset = memd.MemoryDataset
        train_set = Dataset(all_data['trn'], trn_transform)
        test_set = Dataset(all_data['tst'], tst_transform)
        valid_set = Dataset(all_data['val'], tst_transform)

        train_loader = DataLoader(train_set, shuffle=True, num_workers=num_workers, batch_size=batch_size)
        test_loader = DataLoader(test_set, shuffle=False, num_workers=num_workers, batch_size=batch_size)
        valid_loader = DataLoader(valid_set, shuffle=False, num_workers=num_workers, batch_size=batch_size)
        return train_loader, test_loader, valid_loader, num_classes

    if dataset == "cifar100":
        trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True,
                                                 transform=trn_transform)
        testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=tst_transform)
        num_classes = len(np.unique(trainset.targets))
        trn_data = {'x': trainset.data, 'y': trainset.targets}
        tst_data = {'x': testset.data, 'y': testset.targets}
        all_data = memd.get_data(trn_data, tst_data, validation=validation, shuffle_classes=class_order is None,
                                 class_order=class_order)
        Dataset = memd.MemoryDataset
        train_set = Dataset(all_data['trn'], trn_transform)
        test_set = Dataset(all_data['tst'], tst_transform)
        valid_set = Dataset(all_data['val'], tst_transform)

        train_loader = DataLoader(train_set, shuffle=True, num_workers=num_workers, batch_size=batch_size)
        test_loader = DataLoader(test_set, shuffle=False, num_workers=num_workers, batch_size=batch_size)
        valid_loader = DataLoader(valid_set, shuffle=False, num_workers=num_workers, batch_size=batch_size)
        return train_loader, test_loader, valid_loader, num_classes

    else:
        train_dataset, test_dataset, valid_dataset, num_classes = get_dataset(data_path, trn_transform, tst_transform,
                                                                              validation,
                                                                              shuffle_classes=class_order is None,
                                                                              class_order=class_order)

        # print(len(train_dataset))
        # print(len(test_dataset))
        # print(len(valid_dataset))
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                  num_workers=num_workers,
                                  pin_memory=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                                 pin_memory=False)
        valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                                  pin_memory=False)
        return train_loader, test_loader, valid_loader, num_classes


# test
# if __name__ == '__main__':
#     a, b, c, d = get_loaders('cars', 4, 256, validation=0.0)
    # for batch_idx, (images, targets) in enumerate(a):
    #     # print(batch_idx)
    #     pass
