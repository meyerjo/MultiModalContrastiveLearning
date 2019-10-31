import itertools
import os
import re

import numpy as np
from PIL import Image
from torchvision.datasets import VisionDataset
import re

class Washington_RGBD_Dataset(VisionDataset):

    def read_test_instance_ids(self, filename):
        """
        Read the testinstance_ids.txt and get the different trials
        :param filename:
        :return:
        """
        assert(os.path.exists(filename))
        with open(filename, 'r') as f:
            test_instances = f.readlines()

        regex_trial_title = re.compile(r'^\*{6}([^*]+)\*{6}')
        result_dict = {}
        current_key = None
        for test_instance in test_instances:
            # ignore empty ones
            if test_instance.strip() == '':
                continue
            # match the regex
            m = regex_trial_title.search(test_instance)
            # add element to the result_dict
            if m is not None:
                current_key = m.group(1).strip()
                result_dict[current_key] = []
            else:
                result_dict[current_key].append(test_instance.strip())
        return result_dict


    def _get_folders(self, folder):
        """
        Returns all directories in the folder
        :param folder:
        :return:
        """
        folders = os.listdir(folder)
        return [
            f for f in folders if os.path.isdir(os.path.join(folder, f))
        ]

    def _get_files(self, folder):
        """
        Returns all the files in the folder
        :param folder:
        :return:
        """
        folders = os.listdir(folder)
        return [
            f for f in folders if os.path.isfile(os.path.join(folder, f))
        ]

    def _group_files_by_stem(self, files):
        regex_filestem = re.compile(
            r'^([a-z]+_\d+_\d+_\d+_)(([a-z]+)\.(txt|png))$'
        )
        _dict_group_files = {}
        for f in files:
            m = regex_filestem.match(f)
            if m is None:
                print('File does not exist: {}'.format(f))
                continue

            filestem = m.group(1)
            filetype = m.group(3)

            if filestem not in _dict_group_files:
                _dict_group_files[filestem] = {}

            _dict_group_files[filestem][filetype] = f

        number_files = set([len(item.keys()) for key, item in _dict_group_files.items()])
        if len(number_files) != 1:
            raise BaseException('Number of files != 1')
        return _dict_group_files


    def read_all_files(self, root_folder):
        dict_classes = {}
        dir_classes = os.listdir(root_folder)
        dir_classes = [d for d in dir_classes if os.path.isdir(os.path.join(root_folder, d))]
        for _c in dir_classes:
            if _c is not dict_classes:
                dict_classes[_c] = {}
            _full_path_class = os.path.join(root_folder, _c)

            # get the class_sequences `apple_1`, `apple_2`
            _class_sequences = self._get_folders(_full_path_class)

            # get files in folder
            for _seq in _class_sequences:
                _full_seq_path = os.path.join(
                    _full_path_class, _seq
                )
                _files = self._get_files(_full_seq_path)

                # group the files by the filestem
                _group_files = self._group_files_by_stem(_files)

                dict_classes[_c][_seq] = _group_files
        return dict_classes


    def __init__(self, root, train=True,
                 transform=None, target_transform=None,
                 download=False, modality=None):

        super(Washington_RGBD_Dataset, self).__init__(root)
        self.transform = transform
        self.target_transform = target_transform
        self.train = train  # training set or test set

        testinstance_ids_file = os.path.join(root, 'testinstance_ids.txt')
        if not os.path.exists(testinstance_ids_file):
            raise FileNotFoundError('File `testinstance_ids.txt` is not found')
        # read the testinstance ids
        testinstance_ids = self.read_test_instance_ids(testinstance_ids_file)
        if not 'trial 1' in testinstance_ids:
            raise BaseException('`trial 1` missing in testinstance_ids')

        test_instances = testinstance_ids['trial 1']

        # things that have to be filled
        # entry = {'data': [...], 'modality': [...]}
        self.data = []
        self.targets = []
        self.classes = set()

        if download:
            raise NotImplementedError(
                'Download Washington RGB-D data')

        if not os.path.exists(os.path.join(root, 'rgbd-dataset')):
            raise NotADirectoryError('Directory "rgbd-dataset" is missing')

        # get the grouped files
        _ds_path = os.path.join(root, 'rgbd-dataset')
        group_files = self.read_all_files(_ds_path)
        # removed the files from
        for _class, _items in group_files.items():
            _seq_keys = group_files[_class].keys()
            for _seq in _seq_keys:
                if self.train:
                    # in this case we want to remove the test instances
                    if _seq in test_instances:
                        del group_files[_class][_seq]
                else:
                    # in this case we want to keep the test instances
                    if _seq not in test_instances:
                        del group_files[_class][_seq]

        self.classes = list(group_files.keys())

        for _c, _sequences in group_files.items():
            for _seq, data_path in _sequences.items():
                _rgb_path = data_path['crop']
                _depth_path = data_path['depthcrop']
                _mask_path = data_path['maskcrop']

                self.data.append(
                    {
                        'data': [
                            Image.open(os.path.join(_ds_path, _c, _seq, _rgb_path)),
                            Image.open(os.path.join(_ds_path, _c, _seq, _depth_path)),
                            # Image.open(os.path.join(_ds_path, _c, _seq, _mask_path)),
                        ],
                        'modality': [
                            'rgb',
                            'depth',
                            # 'mask'
                        ]
                    }
                )
                self.targets.append(_c)

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

