import argparse

import os
import sys
import time

import torch
import torch.optim as optim
import numpy as np
from torch.optim.lr_scheduler import MultiStepLR

import mixed_precision
from datetime import datetime
from checkpoint import Checkpoint
from datasets import build_dataset, get_dataset, get_encoder_size
from model_baseline import BaselineModel
from stats import StatTracker, AverageMeterSet
from utils import weight_init, _warmup_batchnorm

CURRENT_TIME = lambda: datetime.now().strftime('%Y-%m-%d %H:%M:%S')

parser = argparse.ArgumentParser(
    description='Infomax Representations -- Self-Supervised Training')
# parameters for general training stuff
parser.add_argument('--dataset', type=str, default='STL10')
parser.add_argument('--batch_size', type=int, default=200,
                    help='input batch size (default: 200)')
parser.add_argument('--learning_rate', type=float, default=0.0002,
                    help='learning rate')
parser.add_argument('--seed', type=int, default=1,
                    help='random seed (default: 1)')
parser.add_argument('--amp', action='store_true', default=False,
                    help='Enables automatic mixed precision')

# parameters for model and training objective
parser.add_argument('--classifiers', action='store_true', default=False,
                    help="Wether to run self-supervised encoder or"
                         "classifier training task")
parser.add_argument('--ndf', type=int, default=128,
                    help='feature width for encoder')
parser.add_argument('--n_rkhs', type=int, default=1024,
                    help='number of dimensions in fake RKHS embeddings')
parser.add_argument('--tclip', type=float, default=20.0,
                    help='soft clipping range for NCE scores')
parser.add_argument('--n_depth', type=int, default=3)
parser.add_argument('--use_bn', type=int, default=0)

# parameters for output, logging, checkpointing, etc
parser.add_argument('--output_dir', type=str, default='./runs',
                    help='directory where tensorboard events and checkpoints will be stored')
parser.add_argument('--input_dir', type=str, default='/mnt/imagenet',
                    help="Input directory for the dataset. Not needed For C10,"
                         " C100 or STL10 as the data will be automatically downloaded.")
parser.add_argument('--cpt_load_path', type=str, default='abc.xyz',
                    help='path from which to load checkpoint (if available)')
parser.add_argument('--cpt_name', type=str, default='amdim_cpt.pth',
                    help='name to use for storing checkpoints during training')
parser.add_argument('--run_name', type=str, default='default_run',
                    help='name to use for the tensorbaord summary for this run')

parser.add_argument('--modality', type=str, default='dual',
                    choices=['dual', 'rgb', 'depth'])
parser.add_argument('--modality_to_test', type=str, default='random',
                    choices=['random', 'rgb', 'depth'])
parser.add_argument('--baseline', action='store_true', default=False,
                    help='Indicates whether the whole model should be trained.'
                         'Needs to be combined with classifiers=True')
parser.add_argument('--epochs', type=int, default=None, help='Number of epochs')
# ...
args = parser.parse_args()


def update_train_accuracies(epoch_stats, labels, lgt_glb_mlp):
    '''
    Helper function for tracking accuracy on training set
    '''
    labels_np = labels.cpu().numpy()
    max_lgt_glb_mlp = torch.max(lgt_glb_mlp.data, 1)[1].cpu().numpy()
    for j in range(labels_np.shape[0]):
        if labels_np[j] > -0.1:
            hit_glb_mlp = 1 if (max_lgt_glb_mlp[j] == labels_np[j]) else 0
            epoch_stats.update('train_acc_glb_mlp', hit_glb_mlp, n=1)


def test_model(model, test_loader, device, stats, max_evals=200000,
               feat_selection='random'):
    '''
    Evaluate accuracy on test set
    '''
    # warm up batchnorm stats based on current model
    _warmup_batchnorm(model, test_loader, device, batches=50,
                      train_loader=False, feat_selection=feat_selection)

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
    correct_glb_mlp_top_5 = 0.
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
            # images = images[1]
            # `print('Selecting modality: {}'.format(modalities[ind]))
        else:
            images = images.to(device)
        labels = labels.cpu()
        with torch.no_grad():
            res_dict = model(x1=images, x2=images, class_only=True)
            lgt_glb_mlp = res_dict['class']
        # check classification accuracy
        correct_glb_mlp += get_correct_count(lgt_glb_mlp, labels)
        correct_glb_mlp_top_5 += get_correct_count(lgt_glb_mlp, labels, top_k=5)
        total += labels.size(0)
    acc_glb_mlp = correct_glb_mlp / total
    acc_glb_mlp_top_5 = correct_glb_mlp_top_5 / total
    model.train()
    # record stats in the provided stat tracker
    stats.update('test_acc_glb_mlp', acc_glb_mlp, n=1)
    stats.update('test_acc_glb_mlp_top_5', acc_glb_mlp_top_5, n=1)


def main():
    # create target output dir if it doesn't exist yet
    if not os.path.isdir(args.output_dir):
        os.mkdir(args.output_dir)

    # enable mixed-precision computation if desired
    if args.amp:
        mixed_precision.enable_mixed_precision()

    if args.baseline and not args.classifiers:
        raise BaseException(
            'If you want to train the baseline please also activate --classifiers')
    if args.baseline and args.classifiers:
        args.cpt_name = 'amdim_baseline_cpt.pth'
    if args.modality != 'dual':
        if args.modality_to_test != args.modality:
            raise BaseException(
                'Modality for testing should be the same as for testing {} != {}'.format(
                    args.modality_to_test,
                    args.modality
                ))

    # set the RNG seeds (probably more hidden elsewhere...)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    # get the dataset
    dataset = get_dataset(args.dataset)
    enc_size = get_encoder_size(dataset)

    # get a helper object for tensorboard logging
    log_dir = os.path.join(args.output_dir, args.run_name)
    stat_tracker = StatTracker(log_dir=log_dir)

    # get dataloaders for training and testing
    train_loader, test_loader, num_classes = \
        build_dataset(dataset=dataset,
                      batch_size=args.batch_size,
                      input_dir=args.input_dir,
                      labeled_only=args.classifiers,
                      modality=args.modality)

    torch_device = torch.device('cuda')
    # create new model with random parameters
    model = BaselineModel(ndf=args.ndf, n_classes=num_classes,
                          n_rkhs=args.n_rkhs,
                          tclip=args.tclip, n_depth=args.n_depth,
                          enc_size=enc_size,
                          use_bn=(args.use_bn == 1))
    model.init_weights(init_scale=1.0)
    # restore model parameters from a checkpoint if requested
    checkpoint = Checkpoint(
        model, args.cpt_load_path, args.output_dir, cpt_name=args.cpt_name
    )
    model = model.to(torch_device)

    for mod in model.info_modules:
        # reset params in the evaluation classifiers
        mod.apply(weight_init)

    for mod in model.class_modules:
        # reset params in the evaluation classifiers
        mod.apply(weight_init)


    mods_inf = [m for m in model.info_modules]
    mods_cls = [m for m in model.class_modules]
    mods_to_opt = mods_inf + mods_cls
    # configure optimizer
    optimizer = optim.Adam(
        [{'params': mod.parameters(), 'lr': args.learning_rate} for mod in
         mods_to_opt],
        betas=(0.8, 0.999), weight_decay=1e-5, eps=1e-8)

    ce_loss = torch.nn.CrossEntropyLoss()
    scheduler = MultiStepLR(optimizer, milestones=[60, 90, 120], gamma=0.2)
    epochs = 150
    if args.epochs is not None:
        print('Epochs amount overwritten: {}'.format(args.epochs))
        epochs = args.epochs

    model, optimizer = mixed_precision.initialize(model, optimizer)
    model = model.to(torch_device)
    # ...
    for epoch in range(epochs):

        start_epoch = time.time()
        epoch_stats = AverageMeterSet()
        total_elements = 0
        for _, ((images1, images2), labels, modalities) in enumerate(
                train_loader):
            # get data and info about this minibatch
            images1 = images1.to(torch_device).cuda()
            images2 = images2.to(torch_device).cuda()
            labels = labels.to(torch_device).cuda()
            # run forward pass through model and collect activations
            res_dict = model(
                x1=images1, x2=images2,
                class_only=True, modality='rgb',
                training_all=True
            )
            lgt_glb_mlp = res_dict['class']
            # compute total loss for optimization
            loss = ce_loss(lgt_glb_mlp, labels)
            # do optimizer step for encoder
            optimizer.zero_grad()
            mixed_precision.backward(loss,
                                     optimizer)  # special mixed precision stuff
            optimizer.step()
            # record loss and accuracy on minibatch
            epoch_stats.update('loss', loss.item(), n=1)
            update_train_accuracies(epoch_stats, labels, lgt_glb_mlp)
            total_elements += 1


        if epoch % 100 == 0:
            spu = (time.time() - start_epoch) / total_elements
            print(
                '[{0}] Epoch {1:d}, {2:d} data points -- {3:.4f} sec/dp'
                        .format(CURRENT_TIME(), epoch, total_elements, spu)
            )

        # step learning rate scheduler
        scheduler.step(epoch)

        test_model(model, test_loader, torch_device, epoch_stats,
                   max_evals=500000, feat_selection='rgb')
        epoch_str = epoch_stats.pretty_string(ignore=model.tasks)
        diag_str = '[{0}] * epoch {1:d}\n * {2:s}'.format(CURRENT_TIME(), epoch, epoch_str)
        print(diag_str)
        sys.stdout.flush()
        stat_tracker.record_stats(epoch_stats.averages(epoch, prefix='eval/'))



if __name__ == "__main__":
    print(args)
    main()
