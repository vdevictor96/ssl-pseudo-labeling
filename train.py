import sys
import argparse
import math
import copy
from os.path import join as pjoin

from dataloader import get_cifar10, get_cifar100
from utils import accuracy, alpha_weight, plot

from model.wrn import WideResNet

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from utils import accuracy


def train (model, datasets, dataloaders, modelpath,
          criterion, optimizer, scheduler, validation, test, args):

    model_subpath = 'cifar10' if args.num_classes == 10 else 'cifar100'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    training_loss = 1e8
    validation_loss = 1e8
    test_loss = 1e8

    if validation:
        best_model = {
            'epoch': 0,
            'model_state_dict': copy.deepcopy(model.state_dict()),
            'optimizer_state_dict': copy.deepcopy(optimizer.state_dict()),
            'training_losses': [],
            'validation_losses': [],
            'test_losses': [],
            'model_depth' : args.model_depth,
            'num_classes' : args.num_classes,
            'num_labeled' : args.num_labeled,
            'num_validation' : args.num_validation,
            'model_width' : args.model_width,
            'drop_rate' : args.drop_rate,
        } 
    # access datasets and dataloders
    labeled_dataset = datasets['labeled']
    labeled_loader = dataloaders['labeled']
    unlabeled_loader = dataloaders['unlabeled']
    unlabeled_dataset = datasets['unlabeled']
    if validation:
        validation_dataset = datasets['validation']
        validation_loader = dataloaders['validation']
    if test:
        test_dataset = datasets['test']
        test_loader = dataloaders['test']

    print('Training started')
    print('-' * 20)
    model.train()
    # train
    # STAGE ONE -> epoch < args.t1
    # alpha for pseudolabeled loss = 0, we just train over the labeled data
    # STAGE TWO -> args.t1 <= epoch <= args.t2
    # alpha gets calculated for weighting the pseudolabeled data
    # we train over labeled and pseudolabeled data
    training_losses = []
    validation_losses = []
    test_losses = []
    for epoch in range(args.epoch):
        running_loss = 0.0
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
            
            # unlabeled data is used in Stage 2
            if epoch >= args.epoch_t1:
                try:
                    x_ul, y_ul = next(unlabeled_loader)
                except StopIteration:
                    unlabeled_loader = iter(DataLoader(unlabeled_dataset,
                                                    batch_size=args.train_batch,
                                                    shuffle=True,
                                                    num_workers=args.num_workers))
                    x_ul, _ = next(unlabeled_loader)
                x_ul = x_ul.to(device)
        
                # get the subset of pseudo-labelled data
                model.eval()
                output_ul = model(x_ul)
                target_ul = F.softmax(output_ul, dim=1)
                # TODO change the threshold to argument value
                hot_target_ul = torch.where(target_ul > args.threshold, 1, 0)
                idx, y_pl = torch.where(hot_target_ul == 1)
                x_pl = x_ul[idx]
                x_pl = x_pl.to(device)

            # calculate loss for labelled and pseudo-labelled data (if stage 2) and sum up
            model.train()
            if epoch >= args.epoch_t1:
                n_x_pl = x_pl.size(dim=0)
                output_pl = model(x_pl)
                alpha_w = alpha_weight(args.alpha, args.epoch_t1, args.epoch_t2, epoch)
                pl_loss = 0.0 if (output_pl.size(0) == 0) else criterion(output_pl, y_pl) * alpha_w
            else: 
                n_x_pl = 0
                pl_loss = 0.0

            n_x_l = x_l.size(dim=0)
            output_l = model(x_l)
            l_loss = criterion(output_l, y_l)
            total_loss = (l_loss*n_x_l +  pl_loss*n_x_pl) / (n_x_l + n_x_pl)

            # back propagation
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            running_loss += total_loss.item()
        training_loss = running_loss/(args.iter_per_epoch)
        training_losses.append(training_loss)
        print('Epoch: {} : Train Loss : {:.5f} '.format(
            epoch, training_loss))
        
        # Calculate loss for validation set every epoch
        # Save the best model
        # TODO implement early stopping?
        running_loss = 0.0
        if validation:
            model.eval()
            for x_val, y_val in validation_loader:
                with torch.no_grad():
                    x_val, y_val = x_val.to(device), y_val.to(device)
                    output_val = model(x_val)
                    loss = criterion(output_val, y_val)

                    running_loss += loss.item() * x_val.size(0)

            validation_loss = running_loss / len(validation_dataset)
            validation_losses.append(validation_loss)
            print('Epoch: {} : Validation Loss : {:.5f} '.format(
            epoch, validation_loss))

            if best_model['epoch'] == 0 or validation_loss < best_model['validation_losses'][-1]:
                best_model = {
                    'epoch': epoch,
                    'model_state_dict': copy.deepcopy(model.state_dict()),
                    'optimizer_state_dict': copy.deepcopy(optimizer.state_dict()),
                    'training_losses':  copy.deepcopy(training_losses),
                    'validation_losses': copy.deepcopy(validation_losses),
                    'test_losses': copy.deepcopy(test_losses),
                    'model_depth' : args.model_depth,
                    'num_classes' : args.num_classes,
                    'num_labeled' : args.num_labeled,
                    'num_validation' : args.num_validation,
                    'model_width' : args.model_width,
                    'drop_rate' : args.drop_rate
                }
                torch.save(best_model, pjoin(modelpath, 'best_model_{}_{}.pt'.format(model_subpath, args.num_labeled)))
                print('Best model updated with validation loss : {:.5f} '.format(validation_loss))
        # update learning rate
        scheduler.step()
        print("new lr: ", scheduler.get_lr())
        # Check test error with current model over test dataset
        running_loss = 0.0
        if test:
            total_accuracy = []
            test_loss = 0.0
            model.eval()
            for x_test, y_test in test_loader:
                with torch.no_grad():
                    x_test, y_test = x_test.to(device), y_test.to(device)
                    output_test = model(x_test)                              
                    loss = criterion(output_test, y_test)
                    running_loss += loss.item() * x_test.size(0)
                    acc = accuracy(output_test, y_test)
                    total_accuracy.append(sum(acc))
            test_loss = running_loss / len(test_dataset)
            test_losses.append(test_loss)
            print('Epoch: {} : Test Loss : {:.5f} '.format(
                epoch, test_loss))
            print('Accuracy of the network on test images: %d %%' % (
                sum(total_accuracy)/len(total_accuracy)))

    last_model = {
        'epoch': epoch,
        'model_state_dict': copy.deepcopy(model.state_dict()),
        'optimizer_state_dict': copy.deepcopy(optimizer.state_dict()),
        'training_losses':  copy.deepcopy(training_losses),
        'validation_losses': copy.deepcopy(validation_losses),
        'test_losses': copy.deepcopy(test_losses),
        'model_depth' : args.model_depth,
        'num_classes' : args.num_classes,
        'num_labeled' : args.num_labeled,
        'num_validation' : args.num_validation,
        'model_width' : args.model_width,
        'drop_rate' : args.drop_rate
    }
    torch.save(last_model, pjoin(modelpath, 'last_model_{}_{}.pt'.format(model_subpath, args.num_labeled)))
    if validation:
        # recover better weights from validation
        model.load_state_dict(best_model['model_state_dict'])
    return model
