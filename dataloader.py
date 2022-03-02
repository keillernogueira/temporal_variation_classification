import os
import numpy as np

from skimage import transform

import torch
from torch.utils import data

from data_utils import load_image, create_or_load_statistics, create_distributions_over_classes, \
    normalize_images, data_augmentation


class DataLoader(data.Dataset):
    def __init__(self, mode, data, dataset_input_path, crop_size, stride_crop, output_path, model_name):
        super().__init__()
        assert mode in ['Train', 'Test', 'Test_Full']

        self.mode = mode
        self.model_name = model_name
        self.dataset_input_path = dataset_input_path

        self.crop_size = crop_size
        self.stride_crop = stride_crop

        self.output_path = output_path

        # data
        self.data = data
        if self.model_name == 'pixelwise':
            half_crop = crop_size // 2
            self.data = np.pad(self.data, ((half_crop, half_crop), (half_crop, half_crop), (0, 0)), 'symmetric')
        self.num_channels = self.data.shape[-1]

        # labels
        if self.mode == 'Train':
            self.labels = load_image(os.path.join(self.dataset_input_path, "train.png"))
        else:
            self.labels = load_image(os.path.join(self.dataset_input_path, "test.png"))
        self.num_classes = len(np.unique(self.labels)) - 1
        self.labels[np.where(self.labels == 0)] = 10  # so it can be ignored in the loss
        self.labels = self.labels - 1  # from 0 to 9, being 9 the class to be ignored

        if self.model_name == 'pixelwise':
            # for each and every pixel, we should create a new patch
            self.distrib = np.column_stack(np.where(self.labels != 9))
        else:
            self.distrib = self.make_dataset()
        self.mean, self.std = create_or_load_statistics(self.data, self.distrib, self.crop_size,
                                                        self.stride_crop, self.output_path)

        if len(self.distrib) == 0:
            raise RuntimeError('Found 0 images, please check the data set')

    def make_dataset(self):
        return create_distributions_over_classes(self.labels, self.crop_size, self.stride_crop, self.num_classes)

    def __getitem__(self, index):
        # Reading items from list.
        cur_x, cur_y = self.distrib[index][0], self.distrib[index][1]

        img = self.data[cur_x:cur_x + self.crop_size, cur_y:cur_y + self.crop_size, :]
        if self.model_name == 'pixelwise':
            label = self.labels[cur_x, cur_y]
        else:
            label = self.labels[cur_x:cur_x + self.crop_size, cur_y:cur_y + self.crop_size]

        # Normalization.
        normalize_images(img, self.mean, self.std)

        if self.mode == 'Train':
            if self.model_name == 'pixelwise':
                img, _ = data_augmentation(img)
            else:
                img, label = data_augmentation(img, label)

        img = np.transpose(img, (2, 0, 1))

        # Turning to tensors.
        img = torch.from_numpy(img.copy())
        label = torch.from_numpy(label.copy())

        # Returning to iterator.
        return img.double(), label, cur_x, cur_y

    def __len__(self):
        return len(self.distrib)
