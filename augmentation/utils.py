import torchvision.transforms.functional as TF
from torchvision import transforms

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


class RandomResizedCropMultiple(transforms.RandomResizedCrop):

    def __call__(self, inputs):
        assert(len(inputs) > 0)
        i, j, h, w = self.get_params(inputs[0], self.scale, self.ratio)

        for i, input in enumerate(inputs):
            inputs[i] = TF.resized_crop(input, i, j, h, w, self.size, self.interpolation)
        return inputs