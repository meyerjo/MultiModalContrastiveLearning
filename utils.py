import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init

from mixed_precision import maybe_half


def test_model(model, test_loader, device, stats, max_evals=200000, feat_selection='random'):
    '''
    Evaluate accuracy on test set
    '''
    # warm up batchnorm stats based on current model
    _warmup_batchnorm(model, test_loader, device, batches=50, train_loader=False, feat_selection=feat_selection)

    def get_correct_count(lgt_vals, lab_vals, top_k=1):
        # count how many predictions match the target labels
        max_lgt = torch.topk(lgt_vals.cpu().data, k=top_k)[1]
        if top_k == 1:
            max_lgt = max_lgt.flatten()
            num_correct = (max_lgt == lab_vals).sum().item()
        else:
            labels_reshaped = lab_vals.expand(
                max_lgt.transpose(1, 0).shape).transpose(1, 0)
            topk_comparison = max_lgt == labels_reshaped
            num_correct = torch.any(topk_comparison, axis=1).sum().item()
        return num_correct

    # evaluate model on test_loader
    model.eval()
    correct_glb_mlp = 0.
    correct_glb_lin = 0.
    correct_glb_mlp_top_5 = 0.
    correct_glb_lin_top_5 = 0.
    total = 0.
    for _, (images, labels, modalities) in enumerate(test_loader):
        if total > max_evals:
            break
        if isinstance(images, list):
            images = [img.to(device) for img in images]
            # TODO: see comment below (in def _warmup_batchnorm)
            if feat_selection == 'random':
                ind = np.random.randint(0, len(images))
            elif feat_selection == 'rgb':
                ind = 0
            elif feat_selection == 'depth':
                ind = 1
            else:
                raise BaseException('Unknown feature type')
            images = images[ind]
            #images = images[1]
            #`print('Selecting modality: {}'.format(modalities[ind]))
        else:
            images = images.to(device)
        labels = labels.cpu()
        with torch.no_grad():
            res_dict = model(x1=images, x2=images, class_only=True)
            lgt_glb_mlp, lgt_glb_lin = res_dict['class']
        # check classification accuracy
        correct_glb_mlp += get_correct_count(lgt_glb_mlp, labels)
        correct_glb_lin += get_correct_count(lgt_glb_lin, labels)
        correct_glb_mlp_top_5 += get_correct_count(lgt_glb_mlp, labels, top_k=5)
        correct_glb_lin_top_5 += get_correct_count(lgt_glb_lin, labels, top_k=5)
        total += labels.size(0)
    acc_glb_mlp = correct_glb_mlp / total
    acc_glb_lin = correct_glb_lin / total
    acc_glb_mlp_top_5 = correct_glb_mlp_top_5 / total
    acc_glb_lin_top_5 = correct_glb_lin_top_5 / total
    model.train()
    # record stats in the provided stat tracker
    stats.update('test_acc_glb_mlp', acc_glb_mlp, n=1)
    stats.update('test_acc_glb_lin', acc_glb_lin, n=1)
    stats.update('test_acc_glb_mlp_top_5', acc_glb_mlp_top_5, n=1)
    stats.update('test_acc_glb_lin_top_5', acc_glb_lin_top_5, n=1)


def _warmup_batchnorm(model, data_loader, device, batches=100, train_loader=False, feat_selection=None):
    '''
    Run some batches through all parts of the model to warmup the running
    stats for batchnorm layers.
    '''
    assert(feat_selection is not None)
    model.train()
    for i, (images, _, modalities) in enumerate(data_loader):
        if i == batches:
            break
        if train_loader:
            images = images[0]
        # if only one modality is present this modality is passed to the
        # model as center resized / center cropped version of the image
        # This tests the data representation. In the case of multiple
        # modalities or 'privileged' information this needs to be
        # further investigated. TODO: test various ways to do this selection
        # As an initial test we will select a random modality each time
        # this method is called. As the representation should generalize to
        # both of them.
        if isinstance(images, list):
            images = [img.to(device) for img in images]
            if feat_selection == 'random':
                ind = np.random.randint(0, len(images))
            elif feat_selection == 'rgb':
                ind = 0
            elif feat_selection == 'depth':
                ind = 1
            else:
                raise BaseException('Unknown feature type')
            images = images[ind]
        else:
            images = images.to(device)
        _ = model(x1=images, x2=images, class_only=True)


def flatten(x):
    return x.reshape(x.size(0), -1)


def random_locs_2d(x, k_hot=1):
    '''
    Sample a k-hot mask over spatial locations for each set of conv features
    in x, where x.shape is like (n_batch, n_feat, n_x, n_y).
    '''
    # assume x is (n_batch, n_feat, n_x, n_y)
    x_size = x.size()
    n_batch = x_size[0]
    n_locs = x_size[2] * x_size[3]
    idx_topk = torch.topk(torch.rand((n_batch, n_locs)), k=k_hot, dim=1)[1]
    khot_mask = torch.zeros((n_batch, n_locs)).scatter_(1, idx_topk, 1.)
    rand_locs = khot_mask.reshape((n_batch, 1, x_size[2], x_size[3]))
    rand_locs = maybe_half(rand_locs)
    return rand_locs


def init_pytorch_defaults(m, version='041'):
    '''
    Apply default inits from pytorch version 0.4.1 or 1.0.0.

    pytorch 1.0 default inits are wonky :-(
    '''
    if version == '041':
        # print('init.pt041: {0:s}'.format(str(m.weight.data.size())))
        if isinstance(m, nn.Linear):
            stdv = 1. / math.sqrt(m.weight.size(1))
            m.weight.data.uniform_(-stdv, stdv)
            if m.bias is not None:
                m.bias.data.uniform_(-stdv, stdv)
        elif isinstance(m, nn.Conv2d):
            n = m.in_channels
            for k in m.kernel_size:
                n *= k
            stdv = 1. / math.sqrt(n)
            m.weight.data.uniform_(-stdv, stdv)
            if m.bias is not None:
                m.bias.data.uniform_(-stdv, stdv)
        elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
            if m.affine:
                m.weight.data.uniform_()
                m.bias.data.zero_()
        else:
            assert False
    elif version == '100':
        # print('init.pt100: {0:s}'.format(str(m.weight.data.size())))
        if isinstance(m, nn.Linear):
            init.kaiming_uniform_(m.weight, a=math.sqrt(5))
            if m.bias is not None:
                fan_in, _ = init._calculate_fan_in_and_fan_out(m.weight)
                bound = 1 / math.sqrt(fan_in)
                init.uniform_(m.bias, -bound, bound)
        elif isinstance(m, nn.Conv2d):
            n = m.in_channels
            init.kaiming_uniform_(m.weight, a=math.sqrt(5))
            if m.bias is not None:
                fan_in, _ = init._calculate_fan_in_and_fan_out(m.weight)
                bound = 1 / math.sqrt(fan_in)
                init.uniform_(m.bias, -bound, bound)
        elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
            if m.affine:
                m.weight.data.uniform_()
                m.bias.data.zero_()
        else:
            assert False
    elif version == 'custom':
        # print('init.custom: {0:s}'.format(str(m.weight.data.size())))
        if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
            init.normal_(m.weight.data, mean=1, std=0.02)
            init.constant_(m.bias.data, 0)
        else:
            assert False
    else:
        assert False


def weight_init(m):
    '''
    Usage:
        model = Model()
        model.apply(weight_init)
    '''
    if isinstance(m, nn.Linear):
        init_pytorch_defaults(m, version='041')
    elif isinstance(m, nn.Conv2d):
        init_pytorch_defaults(m, version='041')
    elif isinstance(m, nn.BatchNorm1d):
        init_pytorch_defaults(m, version='041')
    elif isinstance(m, nn.BatchNorm2d):
        init_pytorch_defaults(m, version='041')
    elif isinstance(m, nn.Conv1d):
        init.normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.Conv3d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose1d):
        init.normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose2d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose3d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.BatchNorm3d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.LSTM):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, nn.LSTMCell):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, nn.GRU):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, nn.GRUCell):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)


class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, input_tensor):
        return input_tensor.view(input_tensor.size(0), -1)

