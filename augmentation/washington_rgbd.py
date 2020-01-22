import random

import torchvision.transforms.functional as TF
from torchvision import transforms

from augmentation import rand_augment
from augmentation.utils import MultipleInputsToTensor, AddFirstDimension, \
    ResizeMultiple, CenterCropMultiple, MultipleInputsNormalize, \
    RandomResizedCropMultiple, SpeckleNoise
from augmentation.rand_augment import RandAugmentMultipleModalities, RandAugment

INTERP = 3


NORMALIZATION_PARAMS = {

# rgb mean  [125.9179271  116.55520094 110.44606892]
# rgb std   [71.01960914 72.97451793 74.19717702]
    'RGB': {
        'mean': [130.49668729/255., 122.32002796/255., 119.00002824/255.],
        'std':  [56.86385995/255., 58.41541805/255., 63.3006397/255. ]
    },
    'DEPTH': {
        'mean': [117.96825765/255., 117.96825765/255., 117.96825765/255.],
        'std':  [83.57220836/255., 83.57220836/255., 83.57220836/255.]
    },
}
# TEST distribution is the following
# mean_rgb
# Out[3]: array([126.01805254, 119.00382699, 112.006402  ])
# std_rgb = np.std(concat_elements, axis=0)
# std_rgb
# Out[5]: array([71.10925615, 73.17516656, 74.65919546])

# mean_rgb
# Out[5]: array([102.83441545])
# std_rgb
# Out[6]: array([81.72894529])


class Transforms_Washington_RGBD(object):
    '''
    Transforms_Sun_RGBD dataset, for use with 128x128 full image encoder.
    '''
    def custom_flip(self, inputs):
        if random.random() > .5:
            inputs = [TF.hflip(img) for img in inputs]
        return inputs

    def __init__(self, modality=None, normalizer_mod1=None, normalizer_mod2=None):
        """

        :param modality: Modality
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
            self.flip_lr = transforms.RandomHorizontalFlip(p=0.5)
            rand_crop = \
                transforms.RandomResizedCrop(128, scale=(0.3, 1.0), ratio=(0.7, 1.4),
                                             interpolation=INTERP)

            rand_augmentation = RandAugment(3, 4)
            rand_augmentation_depth = RandAugment(3, 4, rand_augment.augment_list_depth())

            speckle_noise = SpeckleNoise(1, 10e-1)

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

            # create the modality customized
            if modality == 'rgb':
                print('Add transforms specific with RGB')
                self.train_transform = transforms.Compose([
                    rand_crop,
                    rand_augmentation,
                    post_transform
                ])
            elif modality == 'd' or modality == 'depth':
                print('Add transforms specific with Depth')
                self.train_transform = transforms.Compose([
                    rand_crop,
                    rand_augmentation_depth,
                    post_transform
                ])

            self.test_transform = transforms.Compose([
                transforms.Resize(146, interpolation=INTERP),
                transforms.CenterCrop(128),
                post_transform
            ])
        elif (modality == 'dual') or (modality == 'rgb_d'):
            # Handling multiple modalities
            self.flip_lr_multiple = self.custom_flip
            rand_crop_multiple = \
                RandomResizedCropMultiple(
                    128, scale=(0.3, 1.0), ratio=(0.7, 1.4), interpolation=INTERP)

            rand_augment_multiple_modalities = \
                RandAugmentMultipleModalities(3, 4, 0)
            rand_augment_multiple_modalities_depth = \
                RandAugmentMultipleModalities(3, 4, 1, 'depth')

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
            # TODO: add enabler
            self.train_transform = transforms.Compose([
                rand_crop_multiple,
                rand_augment_multiple_modalities,
                rand_augment_multiple_modalities_depth,
                post_transform_multiple
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
