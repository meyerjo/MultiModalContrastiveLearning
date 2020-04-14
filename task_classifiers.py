import sys
import time
from datetime import datetime

import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR

import mixed_precision
from costs import loss_xent
from datasets import Dataset
from stats import AverageMeterSet, update_train_accuracies
from utils import weight_init, test_model

CURRENT_TIME = lambda: datetime.now().strftime('%Y-%m-%d %H:%M:%S')


def _train(model, optimizer, scheduler, checkpointer, epochs, train_loader,
           test_loader, stat_tracker, log_dir, device, modality_to_test,
           baseline_training=False):
    '''
    Training loop to train classifiers on top of an encoder with fixed weights.
    -- e.g., use this for eval or running on new data
    '''
    # If mixed precision is on, will add the necessary hooks into the model and
    # optimizer for half precision conversions
    model, optimizer = mixed_precision.initialize(model, optimizer)
    # ...
    time_start = time.time()
    total_updates = 0
    next_epoch, total_updates = checkpointer.get_current_position(classifier=True)
    for epoch in range(next_epoch, epochs):
        epoch_updates = 0
        epoch_stats = AverageMeterSet()
        for _, ((images1, images2), labels, modalities) in enumerate(train_loader):
            # get data and info about this minibatch
            images1 = images1.to(device)
            images2 = images2.to(device)
            labels = labels.to(device)
            # run forward pass through model and collect activations
            res_dict = model(
                x1=images1, x2=images2,
                class_only=True, modality=modality_to_test,
                training_all=baseline_training
            )
            lgt_glb_mlp, lgt_glb_lin = res_dict['class']
            # compute total loss for optimization
            loss = (loss_xent(lgt_glb_mlp, labels) +
                    loss_xent(lgt_glb_lin, labels))
            if baseline_training:
                loss = loss_xent(lgt_glb_mlp, labels)
            # do optimizer step for encoder
            optimizer.zero_grad()
            mixed_precision.backward(loss, optimizer)  # special mixed precision stuff
            optimizer.step()
            # record loss and accuracy on minibatch
            epoch_stats.update('loss', loss.item(), n=1)
            update_train_accuracies(epoch_stats, labels, lgt_glb_mlp, lgt_glb_lin)
            # shortcut diagnostics to deal with long epochs
            total_updates += 1
            epoch_updates += 1
            if (total_updates % 100) == 0:
                time_stop = time.time()
                spu = (time_stop - time_start) / 100.
                print('[{0}] Epoch {1:d}, {2:d} updates -- {3:.4f} sec/update'
                      .format(CURRENT_TIME(), epoch, epoch_updates, spu))
                time_start = time.time()

        # step learning rate scheduler
        scheduler.step(epoch)
        # record diagnostics
        test_model(model, test_loader, device, epoch_stats, max_evals=500000, feat_selection=modality_to_test)
        epoch_str = epoch_stats.pretty_string(ignore=model.tasks)
        diag_str = '[{0}] {1:d}: {2:s}'.format(CURRENT_TIME(), epoch, epoch_str)
        print(diag_str)
        sys.stdout.flush()
        stat_tracker.record_stats(epoch_stats.averages(epoch, prefix='eval/'))

        checkpointer.update(epoch + 1, total_updates, classifier=True)


def train_classifiers(model, learning_rate, dataset, train_loader,
                      test_loader, stat_tracker, checkpointer, log_dir, device,
                      modality_to_test, baseline_training=False, overwrite_epochs=None,
                      label_proportion=None):
    # retrain the evaluation classifiers using the trained feature encoder
    for mod in model.class_modules:
        # reset params in the evaluation classifiers
        mod.apply(weight_init)
    mods_to_opt = [m for m in model.class_modules]
    # configure optimizer
    optimizer = optim.Adam(
        [{'params': mod.parameters(), 'lr': learning_rate} for mod in mods_to_opt],
        betas=(0.8, 0.999), weight_decay=1e-5, eps=1e-8)
    # configure learning rate schedulers
    if dataset in [Dataset.C10, Dataset.C100, Dataset.STL10]:
        scheduler = MultiStepLR(optimizer, milestones=[80, 110], gamma=0.2)
        epochs = 120
    elif dataset == Dataset.IN128:
        scheduler = MultiStepLR(optimizer, milestones=[15, 25], gamma=0.2)
        epochs = 30
    elif dataset == Dataset.PLACES205:
        scheduler = MultiStepLR(optimizer, milestones=[7, 12], gamma=0.2)
        epochs = 15
    elif dataset == Dataset.FALLINGTHINGS and not baseline_training:
        scheduler = MultiStepLR(optimizer, milestones=[15, 25], gamma=0.2)
        epochs = 30
    elif dataset == Dataset.FALLINGTHINGS_RGB_D and not baseline_training:
        scheduler = MultiStepLR(optimizer, milestones=[15, 25], gamma=0.2)
        epochs = 30
    elif dataset == Dataset.FALLINGTHINGS_RGB_DJET and not baseline_training:
        scheduler = MultiStepLR(optimizer, milestones=[15, 25], gamma=0.2)
        epochs = 30
    elif dataset == Dataset.FALLINGTHINGS and baseline_training:
        scheduler = MultiStepLR(optimizer, milestones=[60, 90, 120], gamma=0.2)
        epochs = 150
    elif dataset == Dataset.SUN_RGBD:
        scheduler = MultiStepLR(optimizer, milestones=[15, 25], gamma=0.2)
        epochs = 30
    else:
        raise NotImplementedError('Unknown dataset type: {}'.format(dataset))
    if overwrite_epochs is not None:
        epochs = overwrite_epochs
    # retrain the model
    _train(
        model, optimizer, scheduler, checkpointer, epochs, train_loader,
        test_loader, stat_tracker, log_dir, device,
        modality_to_test, baseline_training
    )
