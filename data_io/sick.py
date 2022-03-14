import itertools
import json
import os
import re
from pathlib import Path

import numpy as np
from PIL import Image
from torchvision.datasets import VisionDataset
import re

from data_io.utils import scaleit3


class Sick_Data(VisionDataset):

    def __init__(self, root, train=True,
                 transform=None, target_transform=None,
                 download=False, modality=None):
        assert download == False, f"Download of sick data not possible"
        assert os.path.exists(root), f"path does not exist: {root}"
        super(Sick_Data, self).__init__(root)
        self.transform = transform
        self.target_transform = target_transform
        self.train = train  # training set or test set

        self.valid_classes = [
            'bag', 'box'
        ]

        # collecting files
        folders_in_root = [os.path.join(root, l) for l in os.listdir(root)]
        folders_in_root = [_p for _p in folders_in_root if os.path.isdir(_p)]

        print(f"[befor filter] Folders in root directory: {len(folders_in_root)}")

        if self.train:
            folders_in_root = [_p for _p in folders_in_root if re.search('train\/?$', _p)]
        else:
            folders_in_root = [_p for _p in folders_in_root if re.search('test\/?$', _p)]
        # assume that we only have one type of folders left by now
        assert(len(folders_in_root) == 1)
        print(f"[after filter] Folders in root directory: {len(folders_in_root)}")

        self.root = Path(folders_in_root[0])

        class_directories = os.listdir(self.root)
        assert len(class_directories) == len(self.valid_classes), "directory account does not match"

        modalities = ['rgb', 'depth']


        # things that have to be filled
        # entry = {'data': [...], 'modality': [...]}
        self.data = []
        self.targets = []
        for _class_dir in self.valid_classes:
            files_in_modalities = []
            for _mod in modalities:
                _full_path = self.root / _class_dir / _mod
                files_in_modalities.append(os.listdir(_full_path))

            chained_files_per_modality = list(itertools.chain(*files_in_modalities))
            set_modality_files = set(chained_files_per_modality)
            # make sure that for all modalities
            # the same amount of files is available
            assert(all([len(set_modality_files) == len(l) for l in files_in_modalities]))

            # set_modality_files contains all the relevant files for all modalities

            for file in set_modality_files:
                # generate the paths
                _rgb_path = self.root / _class_dir / 'rgb' / file
                _depth_path = self.root / _class_dir / 'depth' / file
                rgb_im, depth_im = None, None
                if modality != 'depth':
                    rgb_im = np.array(Image.open(_rgb_path))
                if modality != 'rgb':
                    depth_im = np.array(Image.open(_depth_path))

                # rgb_im is not None
                if rgb_im is not None:
                    assert len(rgb_im.shape) == 3 and rgb_im.shape[-1] == 3
                    mean_im = np.mean(rgb_im, axis=-1).astype(np.uint8)
                    rgb_im = np.repeat(np.expand_dims(mean_im, -1), 3, axis=-1)

                if modality == 'rgb':
                    assert rgb_im is not None
                    entry = {
                        'data': [rgb_im], 'modality': ['rgb']
                    }
                elif modality == 'depth' or modality == 'd':
                    assert depth_im is not None
                    entry = {
                        'data': [depth_im], 'modality': ['depth']
                    }
                else:
                    assert rgb_im is not None and depth_im is not None
                    entry = {
                        'data': [rgb_im, depth_im],
                        'modality': ['rgb', 'depth']
                    }

                self.data.append(entry)
                self.targets.append(_class_dir)

        # convert the overall classes to a list
        _histc = [np.sum(np.array(self.targets) == c) for c in self.valid_classes]
        _class_hist = dict(zip(self.valid_classes, _histc))
        print('[{}]: Class_hist: {}'.format(
            'Train' if self.train else 'Test', _class_hist
        ))
        self.classes = list(self.valid_classes)
        self.class_to_idx = {_class: i for i, _class in enumerate(self.classes)}
        print('Loaded {} datapoints with {} labels (total: #{})'.format(
            len(self.data), len(self.targets), len(self.classes)))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img_dict, target_str = self.data[index], self.targets[index]
        target = self.class_to_idx[target_str]

        image_per_modality = img_dict['data']

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        pil_images = []
        for im in image_per_modality:
            im = scaleit3(im)
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
            return (pil_images[0], pil_images[1]), target, img_dict['modality']
        else:
            raise NotImplementedError('More than 2 modalities not supported ')

