import random

import torchvision.transforms.functional as TF
from torchvision import transforms

from augmentation import rand_augment
from augmentation.rand_augment import RandAugment, \
    RandAugmentMultipleModalities, RandAugmentDictDepth
from augmentation.utils import MultipleInputsToTensor, AddFirstDimension, \
    ResizeMultiple, CenterCropMultiple, MultipleInputsNormalize, \
    RandomResizedCropMultiple

INTERP = 3
# mu_rgb = np.mean(np.reshape(data, (-1, 3)), axis=0)
# mu_rgb
# array([111.90984871, 94.63495585, 92.1148839])
# std_rgb = np.std(np.reshape(data, (-1, 3)), axis=0)
# std_rgb
# array([69.55312085, 69.86397278, 73.03937326])
# ---
# data = np.asarray(rgb_data)
# mu_depth = np.mean(np.reshape(data, (-1, 3)), axis=0)
# std_depth = np.std(np.reshape(data, (-1, 3)), axis=0)
# mu_depth
# array([127.89837949, 127.89837949, 127.89837949])
# std_depth
# array([73.68372625, 73.68372625, 73.68372625])

NORMALIZATION_PARAMS = {
    'RGB': {
        'mean': [96.93719661/255., 77.62113013/255., 73.83990468/255.],
        'std':  [68.86028832/255., 67.50154136/255., 70.0532389/255. ]
    },
    'DEPTH': { #hha encoded
        'mean': [149.20400399/255., 67.20424082/255., 110.04264934/255.],
        'std':  [57.51121648/255., 37.11818585/255., 49.42439366/255.]
    },
}

class Transforms_NYU_RGBD_Ablation(object):
    '''
    Transforms_NYU_RGBD dataset, for use with 128x128 full image encoder.
    '''
    def custom_flip(self, inputs):
        if random.random() > .5:
            inputs = [TF.hflip(img) for img in inputs]
        return inputs

    def __init__(self, modality=None, normalizer_mod1=None, normalizer_mod2=None, use_randaugment=True, randaugment=None):
        """

        :param modality: Modality
        """
        print('This is the ablation function!!')
        assert(randaugment is not None)
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

            rand_augmentation_depth = RandAugmentDictDepth(1, 4, None, randaugment)

            if modality == 'rgb' and use_randaugment:
                raise NotImplementedError('Do not use ablation function for this')
            elif modality == 'rgb':
                raise NotImplementedError('Do not use ablation function for this')
            elif (modality == 'depth' or modality == 'd') and use_randaugment:
                print('Using rand augmentation...')
                self.train_transform = transforms.Compose([
                    rand_crop, rand_augmentation_depth, post_transform
                ])
            elif modality == 'depth' or modality == 'd':
                raise NotImplementedError('Do not use ablation function for this')
            else:
                raise BaseException('Unknown modality')

            self.test_transform = transforms.Compose([
                transforms.Resize(146, interpolation=INTERP),
                transforms.CenterCrop(128),
                post_transform
            ])
        elif (modality == 'dual') or (modality == 'rgb_d'):
            raise NotImplementedError('Do not use ablation function for this')
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
            if use_randaugment:
                self.train_transform = transforms.Compose([
                    rand_crop_multiple,
                    rand_augment_multiple_modalities,
                    rand_augment_multiple_modalities_depth,
                    post_transform_multiple
                ])
            else:
                raise NotImplementedError('Do not use ablation function for this')

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
