import torch
import matplotlib.pyplot as plt
import numpy as np
from os.path import join as pjoin
from torch.utils.data import DataLoader, Subset
from model.wrn import WideResNet
import torch.nn.functional as F
from torch.utils.data import DataLoader

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


def validation_set(base_dataset, num_validation, num_classes):
    '''
    args: 
        base_dataset : (torch.utils.data.Dataset)
    returns : (torch.utils.data.Dataset) subset 
    Description:
        This function samples even ammount of images from each class
        from the base dataset given up to the size of the validation dataset
    '''
    labels = base_dataset.targets
    label_per_class = num_validation // num_classes
    labels = np.array(labels)
    validation_idx = []
    for i in range(num_classes):
        idx = np.where(labels == i)[0]
        idx = np.random.choice(idx, label_per_class, False)
        validation_idx.extend(idx)
    validation_idx = np.array(validation_idx)
    np.random.shuffle(validation_idx)
    assert len(validation_idx) == num_validation
    return Subset(base_dataset, validation_idx)


def test_accuracy(testdataset, filepath = "./path/to/model.pth.tar"):
    # CREATE LOADER 
   
    test_loader = DataLoader(testdataset,
                             batch_size=64,
                             shuffle=False,
                             num_workers=1)
    
    
    # RETRIEVE MODEL
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    modelpath = torch.load(pjoin(filepath))
    model = WideResNet(modelpath['model_depth'],
                       modelpath['num_classes'], widen_factor=modelpath['model_width'], dropRate=modelpath['drop_rate'])
    model = model.to(device)
    model.load_state_dict(modelpath['model_state_dict'])

    # CALCULATE ACCURACY
    model.eval()
    total_accuracy = []
    for x_test, y_test in test_loader:
        with torch.no_grad():
            x_test, y_test = x_test.to(device), y_test.to(device)
            output_test = model(x_test)
            acc = accuracy(output_test, y_test)
            total_accuracy.append(sum(acc))
    print('Accuracy of the network on test images: %d %%' % (
                sum(total_accuracy)/len(total_accuracy)))
    