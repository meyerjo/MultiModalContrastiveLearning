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


    def __init__(self, modality=None):
        """

        :param modality: Modality
        """
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
                        mean=[0.405, 0.346, 0.293], std=[0.281, 0.274, 0.274]
                    )]
            elif modality == 'd' or modality == 'depth':
                post_transform_steps += [
                    transforms.Normalize(
                        mean=[0.724, 0.324, 0.143], std=[0.332, 0.363, 0.281]
                    )]
            else:
                raise BaseException('Unknown modality')
            # post_transform_steps += [
            #     AddFirstDimension()
            # ]

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
                        [0.405, 0.346, 0.293],
                        [0.724, 0.324, 0.143]
                    ],
                    std=[
                        [0.281, 0.274, 0.274],
                        [0.332, 0.363, 0.281],
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
