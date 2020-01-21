import itertools
import json
import os
import re

import numpy as np
from PIL import Image
from torchvision.datasets import VisionDataset
import re

from data_io.utils import scaleit3


class NYU_RGBD_Dataset(VisionDataset):

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
            r'^([a-z_]*_\d+_\d+_\d+_)(([a-z]+)\.(txt|png))$'
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

        number_dict_keys = [len(item.keys()) for key, item in _dict_group_files.items()]
        number_files = set(number_dict_keys)

        # TODO: verify that his is not required
        # if len(number_files) != 1:
        #     raise BaseException('Number of files != 1')
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
        assert(os.path.exists(root))
        super(NYU_RGBD_Dataset, self).__init__(root)
        self.transform = transform
        self.target_transform = target_transform
        self.train = train  # training set or test set

        self.valid_classes = [
            'bathtub', 'bed', 'bookshelf', 'box', 'chair',
            'counter', 'desk', 'door', 'dresser', 'garbage bin', 'lamp',
            'monitor', 'night stand', 'pillow', 'sink', 'sofa', 'table',
            'television', 'toilet'
        ]

        assert(os.path.exists(os.path.join(root, 'classes.json')))

        with open(os.path.join(root, 'classes.json')) as f:
            classes = json.loads(f.read())

        folders_in_root = [os.path.join(root, l) for l in os.listdir(root)]
        folders_in_root = [_p for _p in folders_in_root if os.path.isdir(_p)]

        if self.train:
            folders_in_root = [_p for _p in folders_in_root if re.search('train\/?$', _p)]
        else:
            folders_in_root = [_p for _p in folders_in_root if re.search('test\/?$', _p)]
        # assume that we only have one type of folders left by now
        assert(len(folders_in_root) == 1)

        data_dir = folders_in_root[0]

        modalities = os.listdir(data_dir)
        assert(len(modalities) > 0)
        # TODO: filter the modalities according to the filtered


        # get the classes
        # TODO: generate a new file which contains all the classes
        # This class is assumed to be
        class_dirs = os.listdir(os.path.join(data_dir, modalities[0]))
        print('Class dirs found: {}'.format(len(class_dirs)))


        # things that have to be filled
        # entry = {'data': [...], 'modality': [...]}
        self.data = []
        self.targets = []
        self.classes = set(classes)

        if download:
            raise NotImplementedError(
                'Download NYU RGB-D')

        for _class_dir in class_dirs:
            # restrict it to 19 classes
            if _class_dir not in self.valid_classes:
                continue

            files_in_modalities = []
            for _mod in modalities:
                _full_path = os.path.join(data_dir, _mod, _class_dir)
                files_in_modalities.append(os.listdir(_full_path))
            chained_files_per_modality = list(itertools.chain(*files_in_modalities))
            set_modality_files = set(chained_files_per_modality)
            # make sure that for all modalities
            # the same amount of files is available
            assert(all([len(set_modality_files) == len(l) for l in files_in_modalities]))

            # set_modality_files contains all the relevant files for all modalities

            for file in set_modality_files:
                # generate the paths
                _rgb_path = os.path.join(data_dir, 'image', _class_dir, file)
                _depth_path = os.path.join(data_dir, 'depth', _class_dir, file)

                if modality == 'rgb':
                    rgb_im = np.array(Image.open(_rgb_path))
                    entry = {
                        'data': [rgb_im], 'modality': ['rgb']
                    }
                elif modality == 'depth' or modality == 'd':
                    depth_im = np.array(Image.open(_depth_path))
                    entry = {
                        'data': [depth_im], 'modality': ['depth']
                    }
                else:
                    rgb_im = np.array(Image.open(_rgb_path))
                    depth_im = np.array(Image.open(_depth_path))
                    entry = {
                        'data': [rgb_im, depth_im],
                        'modality': ['rgb', 'depth']
                    }

                self.data.append(entry)
                self.targets.append(_class_dir)

        # convert the overall classes to a list
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
        img_dict, target = self.data[index], self.targets[index]
        target = self.class_to_idx[target]

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
            return (pil_images[0][0], pil_images[1][0]), target, img_dict['modality']
        else:
            raise NotImplementedError('More than 2 modalities not supported ')

