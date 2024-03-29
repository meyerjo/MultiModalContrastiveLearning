import sys
import time
from datetime import datetime

import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import MultiStepLR
from torchvision.utils import make_grid

import mixed_precision
from utils import test_model
from stats import AverageMeterSet, update_train_accuracies
from datasets import Dataset
from costs import loss_xent

CURRENT_TIME = lambda: datetime.now().strftime('%Y-%m-%d %H:%M:%S')

def _train(model, optim_inf, scheduler_inf, checkpointer, epochs,
           train_loader, test_loader, stat_tracker, log_dir, device, modality_to_test,
           baseline_training=False, label_proportion=None):
    '''
    Training loop for optimizing encoder
    '''
    # make sure that we are not in baseline training mode
    if baseline_training:
        print('Baseline Training is activated')

    # If mixed precision is on, will add the necessary hooks into the model
    # and optimizer for half() conversions
    model, optim_inf = mixed_precision.initialize(model, optim_inf)
    optim_raw = mixed_precision.get_optimizer(optim_inf)
    # get target LR for LR warmup -- assume same LR for all param groups
    for pg in optim_raw.param_groups:
        lr_real = pg['lr']

    # IDK, maybe this helps?
    torch.cuda.empty_cache()

    # prepare checkpoint and stats accumulator
    next_epoch, total_updates = checkpointer.get_current_position()
    fast_stats = AverageMeterSet()
    # run main training loop
    for epoch in range(next_epoch, epochs):
        epoch_stats = AverageMeterSet()
        epoch_updates = 0
        time_start = time.time()

        for _, ((images1, images2), labels, modalities) in enumerate(train_loader):
            # get data and info about this minibatch
            label_ids = None
            # if label_proportion is not None we expect that some labels in
            # the batch are != -1. Thus, we provide labels for those during
            # training. We retrieve the label_ids of these to filter them
            # from the labels tensor
            if label_proportion is not None:
                label_ids = torch.where(labels != -1)[0]

            labels = labels.to(device)
            concat_labels = torch.cat([labels, labels]).to(device)

            images1 = images1.float().to(device)
            images2 = images2.float().to(device)

            # TODO: squeeze operations are a late change to run it on a custom dataset. Maybe not required
            images1 = images1.squeeze()
            images2 = images2.squeeze()

            # run forward pass through model to get global and local features
            if baseline_training:
                if _ == 0:
                    print('modality_to_test: {}, training_all: {}'.format(
                        modality_to_test, baseline_training
                    ))
                res_dict = model(x1=images1, x2=images2, class_only=False, modality=modality_to_test, training_all=baseline_training)
            else:
                res_dict = model(x1=images1, x2=images2, class_only=False)
            lgt_glb_mlp, lgt_glb_lin = res_dict['class']
            # compute costs for all self-supervised tasks
            loss_g2l = sum(res_dict[l] for l in ['g2l_1t5', 'g2l_1t7', 'g2l_5t5'] if res_dict[l] is not None)
            # loss_g2l = (res_dict['g2l_1t5'] +
            #             res_dict['g2l_1t7'] +
            #             res_dict['g2l_5t5'])
            loss_inf = loss_g2l + res_dict['lgt_reg']

            # compute loss for online evaluation classifiers
            if baseline_training and \
                    (modality_to_test == 'rgb' or modality_to_test == 'depth'):
                if _ == 0:
                    print('Using unconcatenated labels')
                if label_ids is not None:
                    if _ == 0:
                        print('Using {} label proportion resulting in {} labels'.format(
                            label_proportion, label_ids.size()
                        ))
                    if label_ids.size()[0] != 0:
                        loss_cls = (loss_xent(lgt_glb_mlp[label_ids], labels[label_ids]) +
                                    loss_xent(lgt_glb_lin[label_ids], labels[label_ids]))
                    else:
                        print('label_ids.size() == 0', label_ids, label_ids.size())
                        loss_cls = torch.tensor(0.)
                else:
                    loss_cls = (loss_xent(lgt_glb_mlp, labels) +
                                loss_xent(lgt_glb_lin, labels))
            else:
                loss_cls = (loss_xent(lgt_glb_mlp, concat_labels) +
                            loss_xent(lgt_glb_lin, concat_labels))

            # do hacky learning rate warmup -- we stop when LR hits lr_real
            if (total_updates < 500):
                lr_scale = min(1., float(total_updates + 1) / 500.)
                for pg in optim_raw.param_groups:
                    pg['lr'] = lr_scale * lr_real

            # reset gradient accumlators and do backprop
            loss_opt = loss_inf + loss_cls
            optim_inf.zero_grad()
            mixed_precision.backward(loss_opt, optim_inf)  # backwards with fp32/fp16 awareness
            optim_inf.step()

            # record loss and accuracy on minibatch
            updated_stats = {
                'loss_inf': loss_inf.item(),
                'loss_cls': loss_cls.item(),
                'loss_g2l': loss_g2l.item(),
                'lgt_reg': res_dict['lgt_reg'].item(),
            }
            if res_dict['g2l_1t5'] is not None:
                updated_stats['loss_g2l_1t5'] = res_dict['g2l_1t5'].item()
            if res_dict['g2l_1t7'] is not None:
                updated_stats['loss_g2l_1t7'] = res_dict['g2l_1t7'].item()
            if res_dict['g2l_5t5'] is not None:
                updated_stats['loss_g2l_5t5'] = res_dict['g2l_5t5'].item()

            epoch_stats.update_dict(updated_stats, n=1)
            if baseline_training and \
                    (modality_to_test == 'rgb' or modality_to_test == 'depth'):
                update_train_accuracies(epoch_stats, labels, lgt_glb_mlp, lgt_glb_lin)
            else:
                update_train_accuracies(epoch_stats, concat_labels, lgt_glb_mlp, lgt_glb_lin)

            # shortcut diagnostics to deal with long epochs
            total_updates += 1
            epoch_updates += 1
            if (total_updates % 100) == 0:
                # IDK, maybe this helps?
                torch.cuda.empty_cache()
                time_stop = time.time()
                spu = (time_stop - time_start) / 100.
                print('[{0}] Epoch {1:d}, {2:d} updates -- {3:.4f} sec/update'
                      .format(CURRENT_TIME(), epoch, epoch_updates, spu))
                time_start = time.time()
            if (total_updates % 500) == 0:
                # record diagnostics
                eval_start = time.time()
                fast_stats = AverageMeterSet()
                test_model(model, test_loader, device, fast_stats, max_evals=100000)
                stat_tracker.record_stats(
                    fast_stats.averages(total_updates, prefix='fast/'))
                eval_time = time.time() - eval_start
                stat_str = fast_stats.pretty_string(ignore=model.tasks)
                stat_str = '-- {0:d} updates, eval_time {1:.2f}: {2:s}'.format(
                    total_updates, eval_time, stat_str)
                print(stat_str)

        # update learning rate
        scheduler_inf.step(epoch)
        test_model(model, test_loader, device, epoch_stats, max_evals=500000, feat_selection=modality_to_test)
        epoch_str = epoch_stats.pretty_string(ignore=model.tasks)
        diag_str = '[{0}] {1:d}: {2:s}'.format(CURRENT_TIME(), epoch, epoch_str)
        print(diag_str)
        sys.stdout.flush()
        stat_tracker.record_stats(epoch_stats.averages(epoch, prefix='costs/'))
        checkpointer.update(epoch + 1, total_updates)


def train_self_supervised(model, learning_rate, dataset, train_loader,
                          test_loader, stat_tracker, checkpointer, log_dir, device,
                          modality_to_test, baseline_training=False, overwrite_epochs=None,
                          label_proportion=None):
    # configure optimizer
    mods_inf = [m for m in model.info_modules]
    mods_cls = [m for m in model.class_modules]
    mods_to_opt = mods_inf + mods_cls
    optimizer = optim.Adam(
        [{'params': mod.parameters(), 'lr': learning_rate} for mod in mods_to_opt],
        betas=(0.8, 0.999), weight_decay=1e-5, eps=1e-8)
    # configure learning rate schedulers for the optimizers
    if dataset in [Dataset.C10, Dataset.C100, Dataset.STL10]:
        scheduler = MultiStepLR(optimizer, milestones=[250, 280], gamma=0.2)
        epochs = 300
    else:
        # best imagenet results use longer schedules...
        # -- e.g., milestones=[60, 90], epochs=100
        scheduler = MultiStepLR(optimizer, milestones=[30, 45], gamma=0.2)
        epochs = 100
    if overwrite_epochs is not None:
        epochs = overwrite_epochs
    # train the model
    _train(model, optimizer, scheduler, checkpointer, epochs,
           train_loader, test_loader, stat_tracker, log_dir, device, modality_to_test,
           baseline_training, label_proportion=label_proportion)
