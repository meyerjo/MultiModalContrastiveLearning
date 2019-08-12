import os
import random
from enum import Enum

from PIL import Image
import numpy as np
import torch
from torchvision import datasets, transforms

import torchvision.transforms.functional as TF
INTERP = 3

class TransformsFallingThings128:

    class RandomResizedCropMultiple(transforms.RandomResizedCrop):

        def __call__(self, inputs):
            assert(len(inputs) > 0)
            i, j, h, w = self.get_params(inputs[0], self.scale, self.ratio)

            for i, input in enumerate(inputs):
                inputs[i] = TF.resized_crop(input, i, j, h, w, self.size, self.interpolation)
            return inputs


    def custom_flip(self, inputs):
        if random.random() > .5:
            inputs = [TF.hflip(img) for img in inputs]
        return inputs


    '''
    ImageNet dataset, for use with 128x128 full image encoder.
    '''
    def __init__(self):
        # image augmentation functions
        self.flip_lr = transforms.RandomHorizontalFlip(p=0.5)
        # rand_crop = \
        #     transforms.RandomResizedCrop(128, scale=(0.3, 1.0), ratio=(0.7, 1.4),
        #                                  interpolation=INTERP)
        #
        # col_jitter = transforms.RandomApply([
        #     transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8)
        # rnd_gray = transforms.RandomGrayscale(p=0.25)
        # self.train_transform = transforms.Compose([
        #     rand_crop,
        #     col_jitter,
        #     rnd_gray,
        #     post_transform
        # ])

        post_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        self.test_transform = transforms.Compose([
            transforms.Resize(146, interpolation=INTERP),
            transforms.CenterCrop(128),
            post_transform
        ])


        # Handling multiple modalities
        self.flip_lr_multiple = self.custom_flip
        rand_crop_multiple = \
            TransformsFallingThings128.RandomResizedCropMultiple(128, scale=(0.3, 1.0), ratio=(0.7, 1.4),
                                         interpolation=INTERP)

        post_transform_multiple = transforms.Compose([
            MultipleInputsToTensor(),
            # TODO: check the normalization values
            MultipleInputsNormalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        self.test_transform_multiple= transforms.Compose([
            ResizeMultiple(146, interpolation=INTERP),
            CenterCropMultiple(128),
            post_transform_multiple
        ])
        self.train_transform_multi_modality = transforms.Compose([
            rand_crop_multiple,
            # col_jitter,
            # rnd_gray,
            post_transform_multiple
        ])

    def __call__(self, inp):
        if len(inp) == 1:
            # inp = self.flip_lr(inp)
            # out1 = self.train_transform(inp)
            # out2 = self.train_transform(inp)
            # return out1, out2
            raise NotImplementedError('Just be sure this is not called...')
        elif len(inp) == 2:
            from PIL import Image
            types = [isinstance(o, Image.Image) for o in inp]
            assert(all(types))
            inp = self.flip_lr_multiple(inp)
            inp = self.train_transform_multi_modality(inp)
            # return inp[0], inp[1]
            return inp[0][None, ...], inp[1][None, ...]
        else:
            raise BaseException('Unknown number of modalities')


class ResizeMultiple(transforms.Resize):
    def __call__(self, input):
        return [TF.resize(inp, self.size, self.interpolation) for inp in input]


class CenterCropMultiple(transforms.CenterCrop):
    def __call__(self, input):
        return [TF.center_crop(inp, self.size) for inp in input]


class MultipleInputsToTensor(transforms.ToTensor):

    def __call__(self, pic):
        """
        Args:
            pic (PIL Image or numpy.ndarray): Image to be converted to tensor.

        Returns:
            Tensor: Converted image.
        """
        assert(isinstance(pic, list))

        return [TF.to_tensor(p) for p in pic]

class MultipleInputsNormalize(transforms.Normalize):
    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.

        Returns:
            Tensor: Normalized Tensor image.
        """
        assert(isinstance(tensor, list))

        return [TF.normalize(t, self.mean, self.std, self.inplace) for t in tensor]
