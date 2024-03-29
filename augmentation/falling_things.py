import random

import torchvision.transforms.functional as TF
from torchvision import transforms

from augmentation import rand_augment
from augmentation.rand_augment import RandAugmentMultipleModalities, RandAugment
from augmentation.utils import MultipleInputsToTensor, AddFirstDimension, \
    ResizeMultiple, CenterCropMultiple, MultipleInputsNormalize, \
    RandomResizedCropMultiple, RandomResizedCropMultipleIndividually

INTERP = 3


NORMALIZATION_PARAMS = {
    'RGB': {
        'mean': [88.10391786 / 255., 77.88267129 / 255., 61.34734314 / 255.],
        'std':  [45.8098147 / 255., 42.98117045 / 255., 39.68807149 / 255.]
    },
    'DEPTH': {
        'mean': [91.09405015 / 255., 91.09405015 / 255., 91.09405015 / 255.],
        'std':  [61.87289913 / 255., 61.87289913 / 255., 61.87289913 / 255.]
    },
    'JET-DEPTH': {
        'mean': [66.41767038 / 255., 104.19884378 / 255., 165.15607047 / 255.],
        'std':  [80.38406076 / 255., 88.75743179 / 255., 85.48735799 / 255.]
    },
}


class TransformsFallingThings128(object):
    '''
    TransformsFallingThings128 dataset, for use with 128x128 full image encoder.
    '''

    def custom_flip(self, inputs):
        if random.random() > .5:
            inputs = [TF.hflip(img) for img in inputs]
        return inputs


    def __init__(self, modality=None, normalizer_mod1=None, normalizer_mod2=None, use_randaugment=False):
        """

        :param modality: Modality
        :param normalizer_mod1:
        :param normalizer_mod2:
        :param use_randaugment:
        """
        # make sure if one is specified both are specified just to avoid confusion
        if (normalizer_mod1 is None and normalizer_mod2 is not None) or \
                (normalizer_mod1 is not None and normalizer_mod2 is None):
            raise BaseException('If you specify one normalizer you have to specify both')

        # make sure they are both dicts if they are specified
        if normalizer_mod1 is not None and not isinstance(normalizer_mod1, dict):
            raise BaseException('normalizer_mod1 is specified but not as dict')
        if normalizer_mod2 is not None and not isinstance(normalizer_mod2, dict):
            raise BaseException('normalizer_mod2 is specified but not as dict')

        if (normalizer_mod1 is not None) and \
                not all([key in normalizer_mod1.keys() for key in ['mean', 'std']]):
            raise BaseException(
                'Key ("mean" or "std") does not exist in normalizer_mod1')
        if (normalizer_mod2 is not None) and \
                not all([key in normalizer_mod2.keys() for key in ['mean', 'std']]):
            raise BaseException(
                'Key ("mean" or "std") does not exist in normalizer_mod2')

        if normalizer_mod1 is None:
            normalizer_mod1 = NORMALIZATION_PARAMS['RGB']

        if normalizer_mod2 is None:
            normalizer_mod2 = NORMALIZATION_PARAMS['DEPTH']

        # image augmentation functions
        if modality is None or modality == 'rgb' or modality == 'd' or modality == 'depth':
            if use_randaugment:
                print('Rand-Augment usage for single modality not supported')
            self.flip_lr = transforms.RandomHorizontalFlip(p=0.5)
            rand_crop = \
                transforms.RandomResizedCrop(128, scale=(0.3, 1.0), ratio=(0.7, 1.4),
                                             interpolation=INTERP)

            post_transform_steps = [transforms.ToTensor()]
            if modality == 'rgb':
                post_transform_steps += [
                    transforms.Normalize(
                        mean=normalizer_mod1['mean'], std=normalizer_mod1['std']
                    )]
            elif modality == 'd' or modality == 'depth':
                post_transform_steps += [
                    transforms.Normalize(
                        mean=normalizer_mod2['mean'], std=normalizer_mod2['std']
                    )]
            else:
                raise BaseException('Unknown modality')

            post_transform = transforms.Compose(post_transform_steps)

            rand_augmentation = RandAugment(3, 4)
            rand_augmentation_depth = RandAugment(2, 4, rand_augment.augment_list_depth())

            if modality == 'rgb' and use_randaugment:
                print('Using rand augmentation...')
                self.train_transform = transforms.Compose([
                    rand_crop, rand_augmentation, post_transform
                ])
            elif modality == 'rgb':
                print('Not using rand augmentation...')
                self.train_transform = transforms.Compose([
                    rand_crop, post_transform
                ])
            elif (modality == 'depth' or modality == 'd') and use_randaugment:
                print('Using rand augmentation...')
                self.train_transform = transforms.Compose([
                    rand_crop, rand_augmentation_depth, post_transform
                ])
            elif modality == 'depth' or modality == 'd':
                print('Not using rand augmentation...')
                self.train_transform = transforms.Compose([
                    rand_crop, post_transform
                ])
            else:
                raise BaseException('Unknown modality')

            self.test_transform = transforms.Compose([
                transforms.Resize(146, interpolation=INTERP),
                transforms.CenterCrop(128),
                post_transform
            ])
        elif (modality == 'dual') or (modality == 'rgb_d'):
            # Handling multiple modalities
            self.flip_lr_multiple = self.custom_flip
            rand_crop_multiple = \
                RandomResizedCropMultipleIndividually(
                    128, scale=(0.3, 1.0), ratio=(0.7, 1.4), interpolation=INTERP)

            post_transform_multiple = transforms.Compose([
                MultipleInputsToTensor(),
                # TODO: check the normalization values
                MultipleInputsNormalize(
                    mean=[
                        normalizer_mod1['mean'], normalizer_mod2['mean']
                    ],
                    std=[
                        normalizer_mod1['std'], normalizer_mod2['std']
                    ]
                ),
                AddFirstDimension()
            ])
            self.test_transform = transforms.Compose([
                ResizeMultiple(146, interpolation=INTERP),
                CenterCropMultiple(128), post_transform_multiple
            ])
            rand_augment_multiple_modalities = \
                RandAugmentMultipleModalities(3, 4, 0)
            rand_augment_multiple_modalities_depth = \
                RandAugmentMultipleModalities(3, 4, 1, 'depth')

            if use_randaugment:
                self.train_transform = transforms.Compose([
                    rand_crop_multiple,
                    rand_augment_multiple_modalities,
                    rand_augment_multiple_modalities_depth,
                    post_transform_multiple
                ])
            else:
                self.train_transform = transforms.Compose([
                    rand_crop_multiple, post_transform_multiple
                ])

        else:
            raise BaseException('Modality unknown')


    def __call__(self, inp):
        from PIL.Image import Image
        if isinstance(inp, Image) or len(inp) == 1:
            inp = self.flip_lr(inp)
            out1 = self.train_transform(inp)
            out2 = self.train_transform(inp)
            return out1, out2
        elif len(inp) == 2:
            from PIL.Image import Image
            assert(all([isinstance(o, Image) for o in inp]))
            inp = self.flip_lr_multiple(inp)
            inp = self.train_transform(inp)
            return inp[0], inp[1]
        else:
            raise BaseException('Unknown number of modalities')
