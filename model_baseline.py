import numpy as np
import numpy as np
import torch
import torch.nn as nn

from mixed_precision import maybe_half
from model import Encoder, MLPClassifier


def has_many_gpus():
    return torch.cuda.device_count() >= 6


class BaselineModel(nn.Module):
    def __init__(self, ndf, n_classes, n_rkhs, tclip=20.,
                 n_depth=3, use_bn=False, enc_size=32):
        super(BaselineModel, self).__init__()
        self.n_rkhs = n_rkhs
        self.tasks = ('1t5', '1t7', '5t5', '5t7', '7t7')
        dummy_batch = torch.zeros((2, 3, enc_size, enc_size))

        # encoder that provides multiscale features
        self.encoder = Encoder(dummy_batch, nc=3, ndf=ndf, n_rkhs=n_rkhs,
                               n_depth=n_depth, enc_size=enc_size,
                               use_bn=use_bn)
        rkhs_1, rkhs_5, _ = self.encoder(dummy_batch)
        # convert for multi-gpu use
        self.encoder = nn.DataParallel(self.encoder)

        self.n_classes = n_classes
        self.block_glb_mlp = \
            MLPClassifier(rkhs_1.size(1), self.n_classes, n_hidden=1024, p=0.2)


        # gather lists of self-supervised and classifier modules
        self.info_modules = [self.encoder.module]
        self.class_modules = [self.block_glb_mlp]

    def init_weights(self, init_scale=1.):
        self.encoder.module.init_weights(init_scale)

    def encode(self, x, use_eval=False):
        '''
        Encode the images in x, with or without grads detached.
        '''
        if use_eval:
            self.eval()
        x = maybe_half(x)
        rkhs_1, rkhs_5, rkhs_7 = self.encoder(x)
        if use_eval:
            self.train()
        return maybe_half(rkhs_1), maybe_half(rkhs_5), maybe_half(rkhs_7)

    def reset_evaluator(self, n_classes=None):
        '''
        Reset the evaluator module, e.g. to apply encoder on new data.
        - evaluator is reset to have n_classes classes (if given)
        '''
        dim_1 = self.evaluator.dim_1
        if n_classes is None:
            self.n_classes = n_classes
        self.block_glb_mlp = \
            MLPClassifier(dim_1, n_classes, n_hidden=1024, p=0.2)
        self.class_modules = [self.block_glb_mlp]
        return self.block_glb_mlp

    def forward(self, x1, x2, class_only=False, modality=None, training_all=False):
        '''
        Input:
          x1 : images from which to extract features -- x1 ~ A(x)
          x2 : images from which to extract features -- x2 ~ A(x)
          class_only : whether we want all outputs for infomax training
        Output:
          res_dict : various outputs depending on the task
        '''
        # dict for returning various values
        res_dict = {}
        # shortcut to encode one image and evaluate classifier
        if modality is None:
            rkhs_1, _, _ = self.encode(x1)
        elif modality == 'rgb':
            rkhs_1, _, _ = self.encode(x1)
        elif modality == 'd' or modality == 'depth':
            rkhs_1, _, _ = self.encode(x2)
        elif modality == 'random':
            x = x1 if np.random.rand(1) >= .5 else x2
            rkhs_1, _, _ = self.encode(x)
        else:
            raise BaseException('Unknown modality {}'.format(modality))

        lgt_glb_mlp = self.block_glb_mlp(rkhs_1)
        res_dict['class'] = lgt_glb_mlp
        return res_dict

