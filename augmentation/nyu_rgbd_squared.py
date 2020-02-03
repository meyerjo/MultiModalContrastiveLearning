import random

import torchvision.transforms.functional as TF
from torchvision import transforms

from augmentation import rand_augment
from augmentation.rand_augment import RandAugment, RandAugmentMultipleModalities
from augmentation.utils import MultipleInputsToTensor, AddFirstDimension, \
    ResizeMultiple, CenterCropMultiple, MultipleInputsNormalize, \
    RandomResizedCropMultiple

INTERP = 3

NORMALIZATION_PARAMS = {
    'RGB': {
        'mean': [98.31225599897965/255., 80.70434430707843/255., 78.42699086851519/255.],
        'std':  [67.82237432246023/255., 66.73408227228204/255., 69.23595392476584/255.]
    },
    'DEPTH': { #hha encoded
        'mean': [121.06895213637566/255., 72.93286029505053/255., 115.28945389561771/255.],
        'std':  [55.4993485164837/255., 40.07532448618766/255., 50.26966884338791/255.]
    },
}
