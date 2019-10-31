import os
import re
from enum import Enum

from PIL import Image
import numpy as np
import torch
from torchvision import datasets, transforms

from augmentation import falling_things
from augmentation import sun_rgbd
from augmentation.falling_things import TransformsFallingThings128
from augmentation.sun_rgbd import Transforms_Sun_RGBD
from data_io.falling_things import Falling_Things_Dataset
from data_io.sun_rgbd import Sun_RGBD_Dataset
from data_io.washington_rgbd import Washington_RGBD_Dataset

INTERP = 3


class Dataset(Enum):
    C10 = 1
    C100 = 2
    STL10 = 3
    IN128 = 4
    PLACES205 = 5
    FALLINGTHINGS = 6
    FALLINGTHINGS_RGB_D = 6
    FALLINGTHINGS_RGB_DJET = 7
    SUN_RGBD = 8
    WASHINGTON = 9


def get_encoder_size(dataset):
    if dataset in [Dataset.C10, Dataset.C100]:
        return 32
    if dataset == Dataset.STL10:
        return 64
    if dataset in [Dataset.IN128, Dataset.PLACES205]:
        return 128
    if dataset == Dataset.FALLINGTHINGS:
        return 128
    if dataset == Dataset.FALLINGTHINGS_RGB_DJET:
        return 128
    if dataset == Dataset.SUN_RGBD:
        return 128
    raise RuntimeError("Couldn't get encoder size, unknown dataset: {}".format(dataset))


def get_dataset(dataset_name):
    try:
        return Dataset[dataset_name.upper()]
    except KeyError as e:
        raise KeyError("Unknown dataset '" + dataset_name + "'. Must be one of "
                       + ', '.join([d.name for d in Dataset]))


class RandomTranslateWithReflect:
    '''
    Translate image randomly

    Translate vertically and horizontally by n pixels where
    n is integer drawn uniformly independently for each axis
    from [-max_translation, max_translation].

    Fill the uncovered blank area with reflect padding.
    '''

    def __init__(self, max_translation):
        self.max_translation = max_translation

    def __call__(self, old_image):
        xtranslation, ytranslation = np.random.randint(-self.max_translation,
                                                       self.max_translation + 1,
                                                       size=2)
        xpad, ypad = abs(xtranslation), abs(ytranslation)
        xsize, ysize = old_image.size

        flipped_lr = old_image.transpose(Image.FLIP_LEFT_RIGHT)
        flipped_tb = old_image.transpose(Image.FLIP_TOP_BOTTOM)
        flipped_both = old_image.transpose(Image.ROTATE_180)

        new_image = Image.new("RGB", (xsize + 2 * xpad, ysize + 2 * ypad))

        new_image.paste(old_image, (xpad, ypad))

        new_image.paste(flipped_lr, (xpad + xsize - 1, ypad))
        new_image.paste(flipped_lr, (xpad - xsize + 1, ypad))

        new_image.paste(flipped_tb, (xpad, ypad + ysize - 1))
        new_image.paste(flipped_tb, (xpad, ypad - ysize + 1))

        new_image.paste(flipped_both, (xpad - xsize + 1, ypad - ysize + 1))
        new_image.paste(flipped_both, (xpad + xsize - 1, ypad - ysize + 1))
        new_image.paste(flipped_both, (xpad - xsize + 1, ypad + ysize - 1))
        new_image.paste(flipped_both, (xpad + xsize - 1, ypad + ysize - 1))

        new_image = new_image.crop((xpad - xtranslation,
                                    ypad - ytranslation,
                                    xpad + xsize - xtranslation,
                                    ypad + ysize - ytranslation))
        return new_image


class TransformsC10:
    '''
    Apply the same input transform twice, with independent randomness.
    '''

    def __init__(self):
        # flipping image along vertical axis
        self.flip_lr = transforms.RandomHorizontalFlip(p=0.5)
        # image augmentation functions
        normalize = transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                                         std=[x / 255.0 for x in [63.0, 62.1, 66.7]])
        col_jitter = transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.2)], p=0.8)
        img_jitter = transforms.RandomApply([
            RandomTranslateWithReflect(4)], p=0.8)
        rnd_gray = transforms.RandomGrayscale(p=0.25)
        # main transform for self-supervised training
        self.train_transform = transforms.Compose([
            img_jitter,
            col_jitter,
            rnd_gray,
            transforms.ToTensor(),
            normalize
        ])
        # transform for testing
        self.test_transform = transforms.Compose([
            transforms.ToTensor(),
            normalize
        ])

    def __call__(self, inp):
        inp = self.flip_lr(inp)
        out1 = self.train_transform(inp)
        out2 = self.train_transform(inp)
        return out1, out2


class TransformsSTL10:
    '''
    Apply the same input transform twice, with independent randomness.
    '''

    def __init__(self):
        # flipping image along vertical axis
        self.flip_lr = transforms.RandomHorizontalFlip(p=0.5)
        normalize = transforms.Normalize(mean=(0.43, 0.42, 0.39), std=(0.27, 0.26, 0.27))
        # image augmentation functions
        col_jitter = transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.2)], p=0.8)
        rnd_gray = transforms.RandomGrayscale(p=0.25)
        rand_crop = \
            transforms.RandomResizedCrop(64, scale=(0.3, 1.0), ratio=(0.7, 1.4),
                                         interpolation=INTERP)

        self.test_transform = transforms.Compose([
            transforms.Resize(70, interpolation=INTERP),
            transforms.CenterCrop(64),
            transforms.ToTensor(),
            normalize
        ])

        self.train_transform = transforms.Compose([
            rand_crop,
            col_jitter,
            rnd_gray,
            transforms.ToTensor(),
            normalize
        ])

    def __call__(self, inp):
        inp = self.flip_lr(inp)
        out1 = self.train_transform(inp)
        out2 = self.train_transform(inp)
        return out1, out2


class TransformsImageNet128:
    '''
    ImageNet dataset, for use with 128x128 full image encoder.
    '''
    def __init__(self):
        # image augmentation functions
        self.flip_lr = transforms.RandomHorizontalFlip(p=0.5)
        rand_crop = \
            transforms.RandomResizedCrop(128, scale=(0.3, 1.0), ratio=(0.7, 1.4),
                                         interpolation=INTERP)
        col_jitter = transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8)
        rnd_gray = transforms.RandomGrayscale(p=0.25)
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
        self.train_transform = transforms.Compose([
            rand_crop,
            col_jitter,
            rnd_gray,
            post_transform
        ])

    def __call__(self, inp):
        inp = self.flip_lr(inp)
        out1 = self.train_transform(inp)
        out2 = self.train_transform(inp)
        return out1, out2


def build_dataset(dataset, batch_size, input_dir=None, labeled_only=False, modality=None, label_proportion=None):

    train_dir, val_dir = _get_directories(dataset, input_dir)

    if dataset.name != 'C10' and \
        not (os.path.exists(train_dir) and os.path.exists(val_dir)):
        raise BaseException('train_dir or val_dir not exists: {}, {}'.format(
            train_dir, val_dir
        ))

    if dataset == Dataset.C10:
        num_classes = 10
        train_transform = TransformsC10()
        test_transform = train_transform.test_transform
        train_dataset = datasets.CIFAR10(root='/tmp/data/',
                                         train=True,
                                         transform=train_transform,
                                         download=True)
        test_dataset = datasets.CIFAR10(root='/tmp/data/',
                                        train=False,
                                        transform=test_transform,
                                        download=True)
    elif dataset == Dataset.C100:
        num_classes = 100
        train_transform = TransformsC10()
        test_transform = train_transform.test_transform
        train_dataset = datasets.CIFAR100(root='/tmp/data/',
                                          train=True,
                                          transform=train_transform,
                                          download=True)
        test_dataset = datasets.CIFAR100(root='/tmp/data/',
                                         train=False,
                                         transform=test_transform,
                                         download=True)
    elif dataset == Dataset.STL10:
        num_classes = 10
        train_transform = TransformsSTL10()
        test_transform = train_transform.test_transform
        train_split = 'train' if labeled_only else 'train+unlabeled'
        train_dataset = datasets.STL10(root='/tmp/data/',
                                       split=train_split,
                                       transform=train_transform,
                                       download=True)
        test_dataset = datasets.STL10(root='/tmp/data/',
                                      split='test',
                                      transform=test_transform,
                                      download=True)
    elif dataset == Dataset.IN128:
        num_classes = 1000
        train_transform = TransformsImageNet128()
        test_transform = train_transform.test_transform
        train_dataset = datasets.ImageFolder(train_dir, train_transform)
        test_dataset = datasets.ImageFolder(val_dir, test_transform)
    elif dataset == Dataset.PLACES205:
        num_classes = 1000
        train_transform = TransformsImageNet128()
        test_transform = train_transform.test_transform
        train_dataset = datasets.ImageFolder(train_dir, train_transform)
        test_dataset = datasets.ImageFolder(val_dir, test_transform)
    elif dataset == Dataset.FALLINGTHINGS or dataset == Dataset.FALLINGTHINGS_RGB_D:
        num_classes = 21
        train_transform = TransformsFallingThings128(
            modality=modality,
            normalizer_mod1=falling_things.NORMALIZATION_PARAMS['RGB'],
            normalizer_mod2=falling_things.NORMALIZATION_PARAMS['DEPTH']
        )
        test_transform = train_transform.test_transform

        if modality == 'rgb':
            file_regex = re.compile('\.(left|right)\.jpg$')
        elif modality == 'd' or modality == 'depth':
            file_regex = re.compile('\.(left|right)\.depth\.png$')
        else:
            file_regex = None

        train_dataset = Falling_Things_Dataset(
            root=train_dir, train=True, transform=train_transform,
            file_filter_regex=file_regex, label_proportion=label_proportion
        )
        test_dataset = Falling_Things_Dataset(
            root=val_dir, train=False, transform=test_transform,
            file_filter_regex=file_regex
        )
    elif dataset == Dataset.FALLINGTHINGS_RGB_DJET:
        num_classes = 21
        train_transform = TransformsFallingThings128(
            modality=modality,
            normalizer_mod1=falling_things.NORMALIZATION_PARAMS['RGB'],
            normalizer_mod2=falling_things.NORMALIZATION_PARAMS['JET-DEPTH']
        )
        test_transform = train_transform.test_transform

        if modality == 'rgb':
            file_regex = re.compile('\.(left|right)\.jpg$')
        elif modality == 'd' or modality == 'depth':
            file_regex = re.compile('\.(left|right)\.colorized\.depth\.png$')
        else:
            file_regex = None

        train_dataset = Falling_Things_Dataset(
            root=train_dir, train=True, transform=train_transform,
            file_filter_regex=file_regex, label_proportion=label_proportion
        )
        test_dataset = Falling_Things_Dataset(
            root=val_dir, train=False, transform=test_transform,
            file_filter_regex=file_regex
        )
    elif dataset == Dataset.SUN_RGBD:
        assert(label_proportion is None)
        num_classes = 45
        train_transform = Transforms_Sun_RGBD(
            modality=modality,
            normalizer_mod1=sun_rgbd.NORMALIZATION_PARAMS['RGB'],
            normalizer_mod2=sun_rgbd.NORMALIZATION_PARAMS['DEPTH']
        )
        test_transform = train_transform.test_transform

        train_dataset = Sun_RGBD_Dataset(
            root=train_dir, train=True,
            transform=train_transform, modality=modality
        )
        test_dataset = Sun_RGBD_Dataset(
            root=val_dir, train=False,
            transform=test_transform, modality=modality
        )
    elif dataset == Dataset.WASHINGTON:
        assert(label_proportion is None)
        print('Updated classes/transforms')
        num_classes = 45
        train_transform = Transforms_Sun_RGBD(
            modality=modality,
            normalizer_mod1=sun_rgbd.NORMALIZATION_PARAMS['RGB'],
            normalizer_mod2=sun_rgbd.NORMALIZATION_PARAMS['DEPTH']
        )
        test_transform = train_transform.test_transform

        train_dataset = Washington_RGBD_Dataset(
            root=train_dir, train=True,
            transform=train_transform, modality=modality
        )
        test_dataset = Washington_RGBD_Dataset(
            root=val_dir, train=False,
            transform=test_transform, modality=modality
        )

    # build pytorch dataloaders for the datasets
    train_loader = \
        torch.utils.data.DataLoader(dataset=train_dataset,
                                    batch_size=batch_size,
                                    shuffle=True,
                                    pin_memory=True,
                                    drop_last=True,
                                    num_workers=16)
    test_loader = \
        torch.utils.data.DataLoader(dataset=test_dataset,
                                    batch_size=batch_size,
                                    shuffle=True,
                                    pin_memory=True,
                                    drop_last=True,
                                    num_workers=16)

    return train_loader, test_loader, num_classes


def _get_directories(dataset, input_dir):
    if dataset in [Dataset.C10, Dataset.C100, Dataset.STL10]:
        # Pytorch will download those datasets automatically
        return None, None
    if dataset == Dataset.IN128:
        train_dir = os.path.join(input_dir, 'ILSVRC2012_img_train/')
        val_dir = os.path.join(input_dir, 'ILSVRC2012_img_val/')
    elif dataset == Dataset.PLACES205:
        train_dir = os.path.join(input_dir, 'places205_256_train/')
        val_dir = os.path.join(input_dir, 'places205_256_val/')
    elif dataset == Dataset.FALLINGTHINGS or \
            dataset == Dataset.FALLINGTHINGS_RGB_DJET or \
            dataset == Dataset.FALLINGTHINGS_RGB_D:
        train_dir = os.path.join(input_dir, 'train')
        val_dir = os.path.join(input_dir, 'val')
    elif dataset == Dataset.SUN_RGBD:
        train_dir = input_dir
        val_dir = input_dir
    elif dataset == Dataset.WASHINGTON:
        train_dir = input_dir
        val_dir = input_dir
    else:
        raise 'Data directories for dataset ' + dataset + ' are not defined'
    return train_dir, val_dir
