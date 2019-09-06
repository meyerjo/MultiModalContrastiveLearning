import random

import torchvision.transforms.functional as TF
from torchvision import transforms

INTERP = 3

class TransformsFallingThings128(object):
    '''
    TransformsFallingThings128 dataset, for use with 128x128 full image encoder.
    '''

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

        if not all([key in normalizer_mod1.keys() for key in ['mean', 'std']]):
            raise BaseException('Key does not exist')
        if not all([key in normalizer_mod2.keys() for key in ['mean', 'std']]):
            raise BaseException('Key does not exist')

        if normalizer_mod1 is None:
            normalizer_mod1 = {
                'mean': [88.10391786 / 255.,  77.88267129 / 255.,  61.34734314 / 255.],
                'std': [45.8098147 / 255., 42.98117045 / 255., 39.68807149 / 255.]
            }

        if normalizer_mod2 is None:
            normalizer_mod2 = {
                'mean': [91.09405015 / 255., 91.09405015 / 255., 91.09405015 / 255.],
                'std': [61.87289913 / 255., 61.87289913 / 255., 61.87289913 / 255.]
            }

        # image augmentation functions
        if modality is None or modality == 'rgb' or modality == 'd' or modality == 'depth':
            self.flip_lr = transforms.RandomHorizontalFlip(p=0.5)
            rand_crop = \
                transforms.RandomResizedCrop(128, scale=(0.3, 1.0), ratio=(0.7, 1.4),
                                             interpolation=INTERP)

            col_jitter = transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8)
            rnd_gray = transforms.RandomGrayscale(p=0.25)


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


            self.train_transform = transforms.Compose([
                rand_crop,
                # col_jitter,
                # rnd_gray,
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
                TransformsFallingThings128.RandomResizedCropMultiple(
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

        return [TF.normalize(t, self.mean[i], self.std[i], self.inplace) for i, t in enumerate(tensor)]

class AddFirstDimension(object):

    def __call__(self, input):
        if isinstance(input, list):
            return [inp[None, ...] for inp in input]
        return input[None, ...]
