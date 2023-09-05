import random
import numpy as np
from PIL import Image
from torch.utils.data import Dataset


class MemoryDataset(Dataset):
    """Characterizes a dataset for PyTorch -- this dataset pre-loads all images in memory"""

    def __init__(self, data, transform):
        """Initialization"""
        self.labels = data['y']
        self.images = data['x']
        self.transform = transform

    def __len__(self):
        """Denotes the total number of samples"""
        return len(self.images)

    def __getitem__(self, index):
        """Generates one sample of data"""
        x = Image.fromarray(self.images[index])
        x = self.transform(x)
        y = self.labels[index]
        return x, y


def get_data(trn_data, tst_data, validation, shuffle_classes, class_order=None):
    """Prepare data: dataset splits, task partition, class order"""

    data = {}
    taskcla = []

    if class_order is None:
        num_classes = len(np.unique(trn_data['y']))
        class_order = list(range(num_classes))
    else:
        num_classes = len(class_order)
        class_order = class_order.copy()
    if shuffle_classes:
        np.random.shuffle(class_order)

        # initialize data structure
        data['trn'] = {'x': [], 'y': []}
        data['val'] = {'x': [], 'y': []}
        data['tst'] = {'x': [], 'y': []}

    # ALL OR TRAIN
    filtering = np.isin(trn_data['y'], class_order)
    if filtering.sum() != len(trn_data['y']):
        trn_data['x'] = trn_data['x'][filtering]
        trn_data['y'] = np.array(trn_data['y'])[filtering]
    for this_image, this_label in zip(trn_data['x'], trn_data['y']):
        # If shuffling is false, it won't change the class number
        this_label = class_order.index(this_label)
        # add it to the corresponding split
        data['trn']['x'].append(this_image)
        data['trn']['y'].append(this_label)

    # ALL OR TEST
    filtering = np.isin(tst_data['y'], class_order)
    if filtering.sum() != len(tst_data['y']):
        tst_data['x'] = tst_data['x'][filtering]
        tst_data['y'] = tst_data['y'][filtering]
    for this_image, this_label in zip(tst_data['x'], tst_data['y']):
        # If shuffling is false, it won't change the class number
        this_label = class_order.index(this_label)
        # add it to the corresponding split
        data['tst']['x'].append(this_image)
        data['tst']['y'].append(this_label)

    data['ncla'] = len(np.unique(data['trn']['y']))

    # validation
    if validation > 0.0:
        for cc in range(data['ncla']):
            cls_idx = list(np.where(np.asarray(data['trn']['y']) == cc)[0])
            rnd_img = random.sample(cls_idx, int(np.round(len(cls_idx) * validation)))
            rnd_img.sort(reverse=True)
            for ii in range(len(rnd_img)):
                data['val']['x'].append(data['trn']['x'][rnd_img[ii]])
                data['val']['y'].append(data['trn']['y'][rnd_img[ii]])
                data['trn']['x'].pop(rnd_img[ii])
                data['trn']['y'].pop(rnd_img[ii])

    # convert them to numpy arrays
    for split in ['trn', 'val', 'tst']:
        data[split]['x'] = np.asarray(data[split]['x'])

    return data
