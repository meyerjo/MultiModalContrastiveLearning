import itertools
import os
import pickle
import re
import sys
import numpy as np
import torch

from PIL import Image
from torchvision.datasets import VisionDataset
from torchvision.datasets.utils import check_integrity


class Falling_Things_Dataset(VisionDataset):

    def __init__(self, root, train=True,
                 transform=None, target_transform=None,
                 download=False, file_filter_regex=None):

        super(Falling_Things_Dataset, self).__init__(root)
        self.transform = transform
        self.target_transform = target_transform

        self.train = train  # training set or test set

        if download:
            raise NotImplementedError('has not been implemented')

        print('Collecting classes...')
        # TODO:  These need to be replace by the actual files in the root dir
        classes_in_root = [f for f in os.listdir(root)]
        print('Collecting all specific files...')
        files_list = []
        for c in classes_in_root:
            files_list += [
                (os.path.join(self.root, c, f), c, f) for f in os.listdir(os.path.join(self.root, c))
                if os.path.isfile(os.path.join(self.root, c, f))
            ]
        if file_filter_regex is not None:
            files_list = [f for f in files_list if file_filter_regex.search(f[0]) is not None]
         
        if len(files_list) == 0:
            raise BaseException('file list is empty')
        print('Grouping files by filestem')
        # Group them by filestem
        # TODO: test this with multiple modalities
        files_dict = dict()
        for (file_path, cl, file_name) in files_list:
            m = re.match('^([^\.]+)\.(.*)\.(jpg|png)$', file_name)
            if m is None:
                continue
            filestem, modality, extension = m.groups()
            # The Image Filestem is not unique so we need to add the class as well
            dict_key = '{}_{}'.format(cl, filestem)
            if dict_key not in files_dict:
                files_dict[dict_key] = {
                    'file_paths': [],
                    'file_names': [],
                    'modalities': [],
                    'class': cl,
                }
            files_dict[dict_key]['file_paths'].append(file_path)
            files_dict[dict_key]['file_names'].append(file_name)
            files_dict[dict_key]['modalities'].append(modality)

        # make sure we have the same number of modality-entries for each element
        if not (len(set([len(item['modalities']) for k, item in files_dict.items()])) == 1):
            raise BaseException('We expect the number of modalities read per entry to be unique across all data points. However, reading resulting in a different amount of modalities: {}'.format(set([len(item['modalities']) for k, item in files_dict.items()])))

        modality_list = [item['modalities'] for k, item in files_dict.items()]
        modality_list_flat = list(itertools.chain(*modality_list))

        print('Found the following modalities: {}'.format(set(modality_list_flat)))
        print('Loading the data...')
        self.data = []
        self.targets = []
        self.classes = set()

        # TODO: how to handle different image sizes
        # now load the picked numpy arrays
        for key, entries in files_dict.items():
            if len(self.data) % 1000 == 0:
                print('...loaded {} data points...'.format(len(self.data)))

            entry = {
                'data': [],
                'modality': []
            }

            ind = np.argsort(entries['modalities'])
            entries['file_paths'] = np.array(entries['file_paths'])[ind].tolist()
            entries['modalities'] = np.array(entries['modalities'])[ind].tolist()

            # make sure that all files have the same filestem
            assert(
                len(
                    set([
                        os.path.splitext(os.path.basename(f))[0].split('.')[0]
                        for f in entries['file_paths']
                    ])
                ) == 1
            )
            for i in range(len(entries['file_paths'])):
                entry['data'] += [np.array(Image.open(entries['file_paths'][i]))]
                entry['modality'] += [entries['modalities'][i]]

            assert(len(entry['modality']) <= 2)
            self.data.append(entry)

            self.targets.append(entries['class'])
            self.classes.add(entries['class'])
        self.classes = list(self.classes)

        self._load_meta()

    def _load_meta(self):
        self.class_to_idx = {_class: i for i, _class in enumerate(self.classes)}

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
            pil_images.append(Image.fromarray(im))

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

    def __len__(self):
        return len(self.data)
