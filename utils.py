import torch
import matplotlib.pyplot as plt
import numpy as np 
from os.path import join as pjoin

def accuracy(output, target, topk=(1,)):
    """
    Function taken from pytorch examples:
    https://github.com/pytorch/examples/blob/master/imagenet/main.py
    Computes the accuracy over the k top predictions for the specified
    values of k
    """
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def alpha_weight(alpha, t1, t2, curr_epoch):
    """ Calculate alpha regulariser
    """
    if curr_epoch < t1:
        return 0.0
    elif curr_epoch > t2:
        return alpha
    else:
        return ((curr_epoch-t1) / (t2-t1))*alpha

def plot(metric, label, color='b'):
    """  Generates a plot of a given metric given along the epochs
    """
    epochs = range(len(metric))
    plt.plot(epochs, metric, color, label=label)
    plt.title(label)
    plt.xticks(np.arange(0, len(epochs), 2.0))
    plt.xlabel('Epochs')
    plt.ylabel(label)
    plt.legend()
    plt.show()

def plot_model(modelpath, attrname, label, color='b'):
    """ Generates a plot of a given attribute from a model
        Training, validation, test loss
    """
    model_cp = torch.load(pjoin(modelpath))
    plot(model_cp[attrname], label, color)
