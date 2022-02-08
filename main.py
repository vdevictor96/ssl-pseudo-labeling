import argparse
import math

from dataloader import get_cifar10, get_cifar100
from utils      import accuracy

from model.wrn  import WideResNet

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader


def alpha_weight(epoch):
    if epoch < args.t1:
        return 0.0
    elif epoch > args.t2:
        return args.alpha
    else:
         return ((epoch-args.t1) / (args.t2-args.t1))*args.alpha

def main(args):
    if args.dataset == "cifar10":
        args.num_classes = 10
        labeled_dataset, unlabeled_dataset, test_dataset = get_cifar10(args, 
                                                                args.datapath)
    if args.dataset == "cifar100":
        args.num_classes = 100
        labeled_dataset, unlabeled_dataset, test_dataset = get_cifar100(args, 
                                                                args.datapath)
    
    args.epoch = math.ceil(args.total_iter / args.iter_per_epoch)
    # TODO should we use arguments or fix it for a certain percentaje of the epochs given?
    # TODO check t2 <= epoch
    args.epoch_t1 = math.ceil(args.t1 / args.iter_per_epoch)
    args.epoch_t2 = math.ceil(args.t2 / args.iter_per_epoch)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    labeled_loader      = iter(DataLoader(labeled_dataset, 
                                    batch_size = args.train_batch, 
                                    shuffle = True, 
                                    num_workers=args.num_workers))
    unlabeled_loader    = iter(DataLoader(unlabeled_dataset, 
                                    batch_size=args.train_batch,
                                    shuffle = True, 
                                    num_workers=args.num_workers))
    test_loader         = DataLoader(test_dataset,
                                    batch_size = args.test_batch,
                                    shuffle = False, 
                                    num_workers=args.num_workers)
    
    model       = WideResNet(args.model_depth, 
                                args.num_classes, widen_factor=args.model_width, dropRate=args.drop_rate)
    model       = model.to(device)

    # TODO add scheduler for momentum
    optimizer = torch.optim.SGD(model.parameters(), lr = args.lr, momentum = args.momentum, weight_decay=args.wd)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.998)
    criterion = nn.CrossEntropyLoss()

    # STAGE ONE -> epoch < args.t1
    # alpha for pseudolabeled loss = 0, we just train over the labeled data
    for epoch in range(args.epoch_t1):
            running_loss = 0
            model.train()
            for i in range(args.iter_per_epoch):
                try:
                    x_l, y_l    = next(labeled_loader)
                except StopIteration:
                    labeled_loader      = iter(DataLoader(labeled_dataset, 
                                                batch_size = args.train_batch, 
                                                shuffle = True, 
                                                num_workers=args.num_workers))
                    x_l, y_l    = next(labeled_loader)
                x_l, y_l    = x_l.to(device), y_l.to(device)

            

                # calculate loss for labeled
                output_l = model(x_l)
                l_loss = criterion(output_l, y_l)
                
                # back propagation
                optimizer.zero_grad()
                l_loss.backward()
                optimizer.step()
                running_loss += l_loss.item()
            print('Epoch: {} : Train Loss : {:.5f} '.format(epoch, running_loss/(args.iter_per_epoch)))

            scheduler.step()

   
    # STAGE TWO -> args.t1 <= epoch <= args.t2
    # alpha gets calculated for weighting the pseudolabeled data
    # we train over labeled and pseudolabeled data
    for epoch in range(args.epoch_t1, args.epoch):
        running_loss = 0
        model.train()
        for i in range(args.iter_per_epoch):
            try:
                x_l, y_l    = next(labeled_loader)
            except StopIteration:
                labeled_loader      = iter(DataLoader(labeled_dataset, 
                                            batch_size = args.train_batch, 
                                            shuffle = True, 
                                            num_workers=args.num_workers))
                x_l, y_l    = next(labeled_loader)
            
            try:
                x_ul, _     = next(unlabeled_loader)
            except StopIteration:
                unlabeled_loader    = iter(DataLoader(unlabeled_dataset, 
                                            batch_size=args.train_batch,
                                            shuffle = True, 
                                            num_workers=args.num_workers))
                x_ul, _     = next(unlabeled_loader)
            
            x_l, y_l    = x_l.to(device), y_l.to(device)
            x_ul        = x_ul.to(device)

           
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
            pl_loss =  0.0 if (output_pl.size(0) == 0) else criterion(output_pl, y_pl)
            output_l = model(x_l)
            l_loss = criterion(output_l, y_l)
            total_loss = (l_loss*n_x_l + alpha_weight(epoch) * pl_loss*n_x_pl) / (n_x_l + n_x_pl)
           
            # back propagation
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            
            running_loss += total_loss.item()
        print('Epoch: {} : Train Loss : {:.5f} '.format(epoch, running_loss/(args.iter_per_epoch)))

        scheduler.step()
   






if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pseudo labeling \
                                        of CIFAR10/100 with pytorch")
    parser.add_argument("--dataset", default="cifar10", 
                        type=str, choices=["cifar10", "cifar100"])
    parser.add_argument("--datapath", default="./data/", 
                        type=str, help="Path to the CIFAR-10/100 dataset")
    parser.add_argument('--num-labeled', type=int, 
                        default=4000, help='Total number of labeled samples')
    parser.add_argument("--lr", default=0.03, type=float, 
                        help="The initial learning rate") 
    parser.add_argument("--momentum", default=0.9, type=float,
                        help="Optimizer momentum")
    parser.add_argument("--wd", default=0.00005, type=float,
                        help="Weight decay")
    parser.add_argument("--expand-labels", action="store_true", 
                        help="expand labels to fit eval steps")
    parser.add_argument('--train-batch', default=64, type=int,
                        help='train batchsize')
    parser.add_argument('--test-batch', default=64, type=int,
                        help='test batchsize')
    parser.add_argument('--total-iter', default=16*3, type=int,
                        help='total number of iterations to run')
    parser.add_argument('--iter-per-epoch', default=16, type=int,
                        help="Number of iterations to run per epoch")
    parser.add_argument('--num-workers', default=1, type=int,
                        help="Number of workers to launch during training")
    parser.add_argument('--threshold', type=float, default=0.95,
                        help='Confidence Threshold for pseudo labeling')
    parser.add_argument("--dataout", type=str, default="./path/to/output/",
                        help="Path to save log files")
    parser.add_argument("--model-depth", type=int, default=28,
                        help="model depth for wide resnet") 
    parser.add_argument("--model-width", type=int, default=2,
                        help="model width for wide resnet")
    parser.add_argument("--alpha", type=int, default=3,
                        help="alpha regulariser for the loss")
    parser.add_argument("--t1", type=int, default=16*1,
                            help="first stage of iterations for calculating the alpha regulariser")
    parser.add_argument("--t2", type=int, default=16*2,
                            help="second stage of iterations for calculating the alpha regulariser")
    parser.add_argument("--drop-rate", type=int, default=0.3,
                            help="drop out rate for wrn")

    # Add more arguments if you need them
    # Describe them in help
    # You can (and should) change the default values of the arguments
    
    args = parser.parse_args()

    main(args)