import sys
import argparse
import math

from dataloader import get_cifar10, get_cifar100
from utils import accuracy

from model.wrn import WideResNet

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader


def train(model, datasets, dataloaders, model_path,
          criterion, optimizer, scheduler, args):

    # calculate alpha regulariser
    def alpha_weight(epoch):
        if epoch < args.t1:
            return 0.0
        elif epoch > args.t2:
            return args.alpha
        else:
            return ((epoch-args.t1) / (args.t2-args.t1))*args.alpha

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # access datasets and dataloders
    labeled_dataset = datasets['labeled']
    unlabeled_dataset = datasets['unlabeled']
    validation_dataset = datasets['validation']
    test_dataset = datasets['test']

    labeled_loader = dataloaders['labeled']
    unlabeled_loader = dataloaders['unlabeled']
    validation_loader = dataloaders['validation']
    test_loader = dataloaders['test']

    # train
    # STAGE ONE -> epoch < args.t1
    # alpha for pseudolabeled loss = 0, we just train over the labeled data
    for epoch in range(args.epoch_t1):
        running_loss = 0
        model.train()
        for i in range(args.iter_per_epoch):
            try:
                x_l, y_l = next(labeled_loader)
            except StopIteration:
                labeled_loader = iter(DataLoader(labeled_dataset,
                                                 batch_size=args.train_batch,
                                                 shuffle=True,
                                                 num_workers=args.num_workers))
                x_l, y_l = next(labeled_loader)
            x_l, y_l = x_l.to(device), y_l.to(device)

            # calculate loss for labeled
            output_l = model(x_l)
            l_loss = criterion(output_l, y_l)

            # back propagation
            optimizer.zero_grad()
            l_loss.backward()
            optimizer.step()
            running_loss += l_loss.item()
        print('Epoch: {} : Train Loss : {:.5f} '.format(
            epoch, running_loss/(args.iter_per_epoch)))

        scheduler.step()

    # STAGE TWO -> args.t1 <= epoch <= args.t2
    # alpha gets calculated for weighting the pseudolabeled data
    # we train over labeled and pseudolabeled data
    for epoch in range(args.epoch_t1, args.epoch):
        running_loss = 0
        model.train()
        for i in range(args.iter_per_epoch):
            try:
                x_l, y_l = next(labeled_loader)
            except StopIteration:
                labeled_loader = iter(DataLoader(labeled_dataset,
                                                 batch_size=args.train_batch,
                                                 shuffle=True,
                                                 num_workers=args.num_workers))
                x_l, y_l = next(labeled_loader)

            try:
                x_ul, _ = next(unlabeled_loader)
            except StopIteration:
                unlabeled_loader = iter(DataLoader(unlabeled_dataset,
                                                   batch_size=args.train_batch,
                                                   shuffle=True,
                                                   num_workers=args.num_workers))
                x_ul, _ = next(unlabeled_loader)

            x_l, y_l = x_l.to(device), y_l.to(device)
            x_ul = x_ul.to(device)

            # get the subset of pseudo-labelled data
            model.eval()
            output_ul = model(x_ul)
            target_ul = F.softmax(output_ul, dim=1)
            # TODO change the threshold to argument value
            hot_target_ul = torch.where(target_ul > 0.52, 1, 0)
            idx, y_pl = torch.where(hot_target_ul == 1)
            x_pl = x_ul[idx]
            x_pl = x_pl.to(device)

            # calculate loss for labelled and pseudo-labelled data and sum up
            model.train()
            n_x_pl = x_pl.size(dim=0)
            n_x_l = x_l.size(dim=0)
            output_pl = model(x_pl)
            pl_loss = 0.0 if (output_pl.size(
                0) == 0) else criterion(output_pl, y_pl)
            output_l = model(x_l)
            l_loss = criterion(output_l, y_l)
            total_loss = (l_loss*n_x_l + alpha_weight(epoch)
                          * pl_loss*n_x_pl) / (n_x_l + n_x_pl)

            # back propagation
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            running_loss += total_loss.item()
        print('Epoch: {} : Train Loss : {:.5f} '.format(
            epoch, running_loss/(args.iter_per_epoch)))

        scheduler.step()
