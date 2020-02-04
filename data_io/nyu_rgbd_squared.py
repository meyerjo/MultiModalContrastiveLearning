import itertools
import json
import os
import re

import numpy as np
from PIL import Image
from torchvision.datasets import VisionDataset
import re

from data_io.nyu_rgbd import NYU_RGBD_Dataset
from data_io.utils import scaleit3


class NYU_RGBD_Dataset_Squared(NYU_RGBD_Dataset):

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img_dict, target = self.data[index], self.targets[index]
        target = self.class_to_idx[target]

        image_per_modality = img_dict['data']

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        pil_images = []
        for im in image_per_modality:
            pil_im = Image.fromarray(im)
            pil_images.append(pil_im)

        if self.transform is not None:
            if len(pil_images) == 1:
                pil_images = self.transform(pil_images[0])
            else:
                pil_images = self.transform(pil_images)

        if self.target_transform is not None:
            target = self.target_transform(target)

        if len(image_per_modality) == 1:
            if isinstance(pil_images, tuple):
                return (pil_images[0], pil_images[1]), target, img_dict['modality']
            return pil_images, target, img_dict['modality']
        elif len(image_per_modality) == 2:
            return (pil_images[0][0], pil_images[1][0]), target, img_dict['modality']
        else:
            raise NotImplementedError('More than 2 modalities not supported ')

