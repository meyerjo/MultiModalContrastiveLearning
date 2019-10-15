import itertools
import os
import re

import numpy as np
from PIL import Image
from torchvision.datasets import VisionDataset


class Sun_RGBD_Dataset(VisionDataset):

    def __init__(self, root, train=True,
                 transform=None, target_transform=None,
                 download=False, file_filter_regex=None):

        super(Sun_RGBD_Dataset, self).__init__(root)
        self.transform = transform
        self.target_transform = target_transform

        self.train = train  # training set or test set

        if download:
            raise NotImplementedError(
                'Download SUN-RGBD from http://rgbd.cs.princeton.edu/')

        if not os.path.exists(os.path.join(root, 'sunrgbd-meta-data')):
            raise NotADirectoryError('Directory "sunrgbd-meta-data" is missing')

        if not os.path.exists(os.path.join(root, 'SUNRGBD')):
            raise NotADirectoryError('Directory "SUNRGBD" is missing')

        if train:
            split_fname = os.path.join(
                root, 'sunrgbd-meta-data', 'sunrgbd_training_images.txt')
        else:
            split_fname = os.path.join(
                root, 'sunrgbd-meta-data', 'sunrgbd_testing_images.txt')
        fold_files = open(split_fname).readlines()
        fold_files = ['/'.join(e.split('/')[:-2]) for e in fold_files]
        fold_files = [f[:-1] if f.endswith('/') else f for f in fold_files]

        self.data = []
        self.targets = []
        self.classes = set()
        for i, img_folder in enumerate(fold_files):
            if i % 500 == 0:
                print('({}/{}) files processed'.format(
                    i, len(fold_files)
                ))

            img_folder = os.path.join(root, img_folder)
            _scene_type_filename = os.path.join(img_folder, 'scene.txt')
            _rgb_folder = os.path.join(img_folder, 'image')
            _depth_folder = os.path.join(img_folder, 'depth')

            if not os.path.exists(_scene_type_filename):
                print('Folder {} does not contain a scene.txt'.format(img_folder))
                continue

            if not os.path.isfile(_scene_type_filename):
                print('scene_type_filename in {} is not a file'.format(
                    _scene_type_filename
                ))
                continue

            if not os.path.exists(_rgb_folder):
                print('Folder {} does not contain a rgb folder'.format(img_folder))
                continue

            if not os.path.exists(_depth_folder):
                print('Folder {} does not contain a depth folder'.format(img_folder))
                continue

            # read the class name and register it to the set of classes
            class_str = open(_scene_type_filename, 'r').read()
            self.classes.add(class_str)
            # get the files
            rgb_files = os.listdir(_rgb_folder)
            depth_files = os.listdir(_depth_folder)
            # filter for files only
            rgb_files = [os.path.join(_rgb_folder, e) for e in rgb_files if os.path.isfile(os.path.join(_rgb_folder, e))]
            depth_files = [os.path.join(_depth_folder, e) for e in depth_files if os.path.isfile(os.path.join(_depth_folder, e))]

            if len(rgb_files) != 1:
                print('Unexpected number of files found in: {} #Files: {}'.format(
                    _rgb_folder, len(rgb_files)))
                continue

            if len(depth_files) != 1:
                print('Unexpected number of files found in: {} #Files: {}'.format(
                    _depth_folder, len(depth_files)))
                continue

            rgb_data = np.array(Image.open(rgb_files[0]))
            depth_data = np.expand_dims(np.array(Image.open(depth_files[0])), 2)
            depth_data = np.repeat(depth_data, repeats=3, axis=2)
            depth_data = depth_data.astype(np.uint8)

            entry = {
                'data': [
                    rgb_data,
                    depth_data
                ],
                'modality': [
                    'rgb',
                    'depth'
                ]
            }

            # add the data to the list
            self.data.append(entry)
            self.targets.append(class_str)

        if len(self.data) != len(self.targets):
            raise BaseException('Unequal number of data / target points')

        # convert the overall classes to a list
        self.classes = list(self.classes)
        self.class_to_idx = {_class: i for i, _class in enumerate(self.classes)}

    def __len__(self):
        return len(self.data)

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

