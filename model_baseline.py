import numpy as np
import torch
import torch.nn as nn

from mixed_precision import maybe_half
from model import MLPClassifier, ConvResNxN, MaybeBatchNorm2d, Conv3x3, \
    ConvResBlock, FakeRKHSConvNet, NopNet


def has_many_gpus():
    return torch.cuda.device_count() >= 6


class BaselineEncoder(nn.Module):
    def __init__(self, dummy_batch, nc=3, ndf=64, n_rkhs=512, n_depth=3,
                 enc_size=32, use_bn=False):
        super(BaselineEncoder, self).__init__()
        self.nc = nc
        self.ndf = ndf
        self.n_rkhs = n_rkhs
        self.use_bn = use_bn
        self.dim2layer = None

        # encoding block for local features
        print('Using a {enc_size}x{enc_size} encoder'.format(enc_size=enc_size))
        if enc_size == 32:
            self.layer_list = nn.ModuleList([
                Conv3x3(nc, ndf, 3, 1, 0, False),
                ConvResNxN(ndf, ndf, 1, 1, 0, use_bn),
                ConvResBlock(ndf * 1, ndf * 2, 4, 2, 0, n_depth, use_bn),
                ConvResBlock(ndf * 2, ndf * 4, 2, 2, 0, n_depth, use_bn),
                MaybeBatchNorm2d(ndf * 4, True, use_bn),
                ConvResBlock(ndf * 4, ndf * 4, 3, 1, 0, n_depth, use_bn),
                ConvResBlock(ndf * 4, ndf * 4, 3, 1, 0, n_depth, use_bn),
                ConvResNxN(ndf * 4, n_rkhs, 3, 1, 0, use_bn),
                MaybeBatchNorm2d(n_rkhs, True, True)
            ])
        elif enc_size == 64:
            self.layer_list = nn.ModuleList([
                Conv3x3(nc, ndf, 3, 1, 0, False),
                ConvResBlock(ndf * 1, ndf * 2, 4, 2, 0, n_depth, use_bn),
                ConvResBlock(ndf * 2, ndf * 4, 4, 2, 0, n_depth, use_bn),
                ConvResBlock(ndf * 4, ndf * 8, 2, 2, 0, n_depth, use_bn),
                MaybeBatchNorm2d(ndf * 8, True, use_bn),
                ConvResBlock(ndf * 8, ndf * 8, 3, 1, 0, n_depth, use_bn),
                ConvResBlock(ndf * 8, ndf * 8, 3, 1, 0, n_depth, use_bn),
                ConvResNxN(ndf * 8, n_rkhs, 3, 1, 0, use_bn),
                MaybeBatchNorm2d(n_rkhs, True, True)
            ])
        elif enc_size == 128:
            self.layer_list = nn.ModuleList([
                Conv3x3(nc, ndf, 5, 2, 2, False, pad_mode='reflect'),
                Conv3x3(ndf, ndf, 3, 1, 0, False),
                ConvResBlock(ndf * 1, ndf * 2, 4, 2, 0, n_depth, use_bn),
                ConvResBlock(ndf * 2, ndf * 4, 4, 2, 0, n_depth, use_bn),
                ConvResBlock(ndf * 4, ndf * 8, 2, 2, 0, n_depth, use_bn),
                MaybeBatchNorm2d(ndf * 8, True, use_bn),
                ConvResBlock(ndf * 8, ndf * 8, 3, 1, 0, n_depth, use_bn),
                ConvResBlock(ndf * 8, ndf * 8, 3, 1, 0, n_depth, use_bn),
                ConvResNxN(ndf * 8, n_rkhs, 3, 1, 0, use_bn),
                MaybeBatchNorm2d(n_rkhs, True, True)
            ])
        else:
            raise RuntimeError("Could not build encoder."
                               "Encoder size {} is not supported".format(enc_size))
        self._config_modules(dummy_batch, [1, 5, 7], n_rkhs, use_bn)

    def init_weights(self, init_scale=1.):
        '''
        Run custom weight init for modules...
        '''
        for layer in self.layer_list:
            if isinstance(layer, (ConvResNxN, ConvResBlock)):
                layer.init_weights(init_scale)
        for layer in self.modules():
            if isinstance(layer, (ConvResNxN, ConvResBlock)):
                layer.init_weights(init_scale)
            if isinstance(layer, FakeRKHSConvNet):
                layer.init_weights(init_scale)

    def _config_modules(self, x, rkhs_layers, n_rkhs, use_bn):
        '''
        Configure the modules for extracting fake rkhs embeddings for infomax.
        '''
        enc_acts = self._forward_acts(x)
        self.dim2layer = {}
        for i, h_i in enumerate(enc_acts):
            for d in rkhs_layers:
                if h_i.size(2) == d:
                    self.dim2layer[d] = i
        # get activations and feature sizes at different layers
        self.ndf_1 = enc_acts[self.dim2layer[1]].size(1)
        self.ndf_5 = enc_acts[self.dim2layer[5]].size(1)
        self.ndf_7 = enc_acts[self.dim2layer[7]].size(1)
        # configure modules for fake rkhs embeddings
        self.rkhs_block_1 = NopNet()
        self.rkhs_block_5 = FakeRKHSConvNet(self.ndf_5, n_rkhs, use_bn)
        self.rkhs_block_7 = FakeRKHSConvNet(self.ndf_7, n_rkhs, use_bn)

    def _forward_acts(self, x):
        '''
        Return activations from all layers.
        '''
        # run forward pass through all layers
        layer_acts = [x]
        for _, layer in enumerate(self.layer_list):
            layer_in = layer_acts[-1]
            layer_out = layer(layer_in)
            layer_acts.append(layer_out)
        # remove input from the returned list of activations
        return_acts = layer_acts[1:]
        return return_acts

    def forward(self, x):
        '''
        Compute activations and Fake RKHS embeddings for the batch.
        '''
        # compute activations in all layers for x
        acts = self._forward_acts(x)
        # gather rkhs embeddings from certain layers
        r1 = self.rkhs_block_1(acts[self.dim2layer[1]])
        r5 = self.rkhs_block_5(acts[self.dim2layer[5]])
        r7 = self.rkhs_block_7(acts[self.dim2layer[7]])
        return r1, r5, r7

class BaselineModel(nn.Module):
    def __init__(self, ndf, n_classes, n_rkhs, tclip=20.,
                 n_depth=3, use_bn=False, enc_size=32):
        super(BaselineModel, self).__init__()
        self.n_rkhs = n_rkhs
        self.tasks = ('1t5', '1t7', '5t5', '5t7', '7t7')
        dummy_batch = torch.zeros((2, 3, enc_size, enc_size))

        # encoder that provides multiscale features
        self.encoder = BaselineEncoder(dummy_batch, nc=3, ndf=ndf, n_rkhs=n_rkhs,
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

